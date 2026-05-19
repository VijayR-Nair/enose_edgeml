#!/usr/bin/env python3
"""
Smooth synchronized sensor traces and extract peak features.

This script is intended to run on the `.npy` sensor segments produced by the
previous GC/sensor synchronization pipeline. For every matching `.npy` file, it:

1. Loads the synchronized sensor array.
2. Smooths each sensor channel with a Butterworth low-pass filter.
3. Detects the main response peak using strict detection first, then relaxed
   detection if needed.
4. Estimates the baseline around the peak.
5. Extracts peak metrics such as height, duration, area, half-height width,
   and 10% asymmetry.
6. Saves all extracted metrics to a CSV file.

Expected input array format:
    column 0: timestamp, usually in nanoseconds
    column 1: sensor 0
    column 2: sensor 1
    column 3: sensor 2
    column 4: sensor 3

Example:
    python smooth_extract_features.py \
        --input-folder "/path/to/synchronized_sensor_npy" \
        --output-csv "/path/to/features.csv" \
        --repeat 1 \
        --min-fd 4 \
        --max-fd 1024
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences


# =============================================================================
# Default processing parameters
# =============================================================================

# Butterworth smoothing settings.
DEFAULT_BUTTER_ORDER = 2
DEFAULT_CUTOFF_HZ = 25.0

# Ignore peaks that occur before this time. This avoids selecting early artifacts.
DEFAULT_MIN_PEAK_TIME_S = 0.45

# Strict peak-detection settings.
DEFAULT_MIN_WIDTH_SEC = 0.05
DEFAULT_MIN_DIST_SEC = 0.08
DEFAULT_PROM_NOISE_K = 1.5

# Relaxed detection settings used only when strict detection finds no peak.
DEFAULT_RELAX_PROM_K = 0.8
DEFAULT_RELAX_W_MULT = 0.7
DEFAULT_RELAX_D_MULT = 0.7
DEFAULT_ABS_PROM_FRAC = 0.0005
DEFAULT_RELAX_ABS_FRAC = 0.0002

# Minimum signal-to-noise ratio used as a guardrail for relaxed/fallback peaks.
DEFAULT_SNR_MIN = 1.2


# =============================================================================
# Small helper functions
# =============================================================================

def robust_noise_mad(x: np.ndarray) -> float:
    """
    Estimate noise using MAD: median absolute deviation.

    The factor 1.4826 makes MAD comparable to standard deviation for normally
    distributed noise, while still being robust to outliers and peaks.
    """
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    return float(1.4826 * np.median(np.abs(x - median)))


def butter_smooth(
    y: np.ndarray,
    dt: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    butter_order: int = DEFAULT_BUTTER_ORDER,
) -> np.ndarray:
    """
    Smooth a signal using a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    y:
        Raw sensor signal.
    dt:
        Sampling interval in seconds.
    cutoff_hz:
        Low-pass cutoff frequency in Hz.
    butter_order:
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Smoothed signal with the same length as the input.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    # Sampling frequency.
    fs = 1.0 / dt

    # Nyquist frequency is fs / 2. Keep cutoff safely below Nyquist.
    nyquist = fs / 2.0
    safe_cutoff = min(cutoff_hz, 0.45 * nyquist)
    normalized_cutoff = max(1e-6, safe_cutoff / nyquist)

    b, a = butter(butter_order, normalized_cutoff, btype="low")
    return filtfilt(b, a, y)


def crossing_time_linear(
    t: np.ndarray,
    y: np.ndarray,
    level: float,
    i1: int,
    i2: int,
) -> float:
    """
    Estimate the time where y crosses `level` between two neighboring points.
    """
    y1, y2 = y[i1], y[i2]

    if y2 == y1:
        return float(t[i1])

    return float(t[i1] + (level - y1) * (t[i2] - t[i1]) / (y2 - y1))


def find_crossing_left(
    t: np.ndarray,
    y: np.ndarray,
    level: float,
    i_start: int,
    i_end: int,
) -> float:
    """
    Search left of the peak for the last crossing of a given signal level.
    """
    for i in range(i_end, i_start, -1):
        y_prev = y[i - 1] - level
        y_cur = y[i] - level

        if y_cur == 0:
            return float(t[i])

        if y_prev == 0 or y_prev * y_cur < 0:
            return crossing_time_linear(t, y, level, i - 1, i)

    return np.nan


def find_crossing_right(
    t: np.ndarray,
    y: np.ndarray,
    level: float,
    i_start: int,
    i_end: int,
) -> float:
    """
    Search right of the peak for the first crossing of a given signal level.
    """
    for i in range(i_start, i_end):
        y_cur = y[i] - level
        y_next = y[i + 1] - level

        if y_cur == 0:
            return float(t[i])

        if y_next == 0 or y_cur * y_next < 0:
            return crossing_time_linear(t, y, level, i, i + 1)

    return np.nan


def empty_feature_result(mode: str = "none") -> Dict[str, float | str]:
    """
    Return a complete feature dictionary filled with NaN values.

    This keeps the final CSV columns consistent even when no valid peak is found.
    """
    return {
        "mode": mode,
        "peak_idx": np.nan,
        "lb_idx": np.nan,
        "rb_idx": np.nan,
        "t_peak_s": np.nan,
        "t_lb_s": np.nan,
        "t_rb_s": np.nan,
        "peak_height": np.nan,
        "duration_sec": np.nan,
        "area": np.nan,
        "trim_start_s": np.nan,
        "width_half_sec": np.nan,
        "t_left_half_s": np.nan,
        "t_right_half_s": np.nan,
        "asym_10": np.nan,
        "snr": np.nan,
    }


# =============================================================================
# Single sensor-channel processing
# =============================================================================

def process_sensor_channel(
    arr: np.ndarray,
    sensor_col: int,
    timestamp_scale: float = 1e-9,
    butter_order: int = DEFAULT_BUTTER_ORDER,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    min_peak_time_s: float = DEFAULT_MIN_PEAK_TIME_S,
    min_width_sec: float = DEFAULT_MIN_WIDTH_SEC,
    min_dist_sec: float = DEFAULT_MIN_DIST_SEC,
    prom_noise_k: float = DEFAULT_PROM_NOISE_K,
    relax_prom_k: float = DEFAULT_RELAX_PROM_K,
    relax_w_mult: float = DEFAULT_RELAX_W_MULT,
    relax_d_mult: float = DEFAULT_RELAX_D_MULT,
    abs_prom_frac: float = DEFAULT_ABS_PROM_FRAC,
    relax_abs_frac: float = DEFAULT_RELAX_ABS_FRAC,
    snr_min: float = DEFAULT_SNR_MIN,
) -> Dict[str, float | str]:
    """
    Smooth one sensor channel and extract the main peak features.

    The detection logic follows the original code:
    strict peak search → relaxed peak search → fallback to global maximum.
    """
    if arr.ndim != 2 or arr.shape[1] <= sensor_col:
        result = empty_feature_result("invalid_array")
        return result

    if arr.shape[0] < 5:
        result = empty_feature_result("too_few_points")
        return result

    # Convert timestamps to seconds and make them relative to the start.
    # The previous pipeline usually stores timestamps in nanoseconds, so the
    # default scale is 1e-9. Change --timestamp-scale if your files differ.
    t = arr[:, 0].astype(float) * timestamp_scale
    t = t - t[0]
    y = arr[:, sensor_col].astype(float)

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return empty_feature_result("bad_timestamps")

    # Smooth the raw signal before peak detection and metric extraction.
    y_smooth = butter_smooth(
        y,
        dt=dt,
        cutoff_hz=cutoff_hz,
        butter_order=butter_order,
    )

    # Ignore the first part of the trace to avoid injection/synchronization noise.
    start_idx = int(round(min_peak_time_s / dt))
    start_idx = min(max(start_idx, 0), len(t) - 1)
    y_trimmed = y_smooth[start_idx:]

    if len(y_trimmed) < 5:
        result = empty_feature_result("too_short_after_trim")
        result["trim_start_s"] = float(t[start_idx])
        return result

    # Estimate noise and overall range on the trimmed signal.
    noise = robust_noise_mad(y_trimmed - np.median(y_trimmed))
    signal_range = float(np.ptp(y_trimmed) + 1e-12)

    def try_detect(
        width_sec: float,
        distance_sec: float,
        prominence_noise_factor: float,
        absolute_prominence_fraction: float,
    ):
        """Run scipy.find_peaks using time-based width/distance settings."""
        min_width_points = max(3, int(width_sec / dt))
        min_distance_points = max(5, int(distance_sec / dt))
        min_prominence = max(
            prominence_noise_factor * noise,
            absolute_prominence_fraction * signal_range,
        )

        return find_peaks(
            y_trimmed,
            prominence=min_prominence,
            width=min_width_points,
            distance=min_distance_points,
        )

    # First try strict detection.
    peaks_rel, props = try_detect(
        min_width_sec,
        min_dist_sec,
        prom_noise_k,
        abs_prom_frac,
    )
    used_mode = "strict"

    # If no peak is found, use relaxed thresholds.
    if len(peaks_rel) == 0:
        peaks_rel, props = try_detect(
            min_width_sec * relax_w_mult,
            min_dist_sec * relax_d_mult,
            prom_noise_k * relax_prom_k,
            relax_abs_frac,
        )
        used_mode = "relaxed"

    global_peak_idx: Optional[int] = None
    fallback_used = False
    final_snr = np.nan

    # Select the strongest detected peak based on prominence.
    if len(peaks_rel) > 0:
        strongest_peak_position = int(np.argmax(props["prominences"]))
        peak_rel = int(peaks_rel[strongest_peak_position])
        peak_global = start_idx + peak_rel

        final_snr = float(
            (y_trimmed[peak_rel] - np.median(y_trimmed)) / (noise + 1e-12)
        )

        # Strict detections are accepted directly. Relaxed detections must pass SNR.
        if used_mode == "strict" or final_snr >= snr_min:
            global_peak_idx = peak_global

    # Fallback: if detection failed, use the maximum value after trim if SNR is OK.
    if global_peak_idx is None:
        peak_rel = int(np.argmax(y_trimmed))
        final_snr = float(
            (y_trimmed[peak_rel] - np.median(y_trimmed)) / (noise + 1e-12)
        )

        if final_snr >= snr_min:
            global_peak_idx = start_idx + peak_rel
            fallback_used = True

    if global_peak_idx is None:
        result = empty_feature_result("none")
        result["trim_start_s"] = float(t[start_idx])
        result["snr"] = float(final_snr) if np.isfinite(final_snr) else np.nan
        return result

    # Estimate left and right baseline positions using peak prominence bases.
    # wlen controls the local window for the prominence calculation.
    wlen = int(max(11, min(len(y_smooth) - 1, 0.8 / dt)))
    _, left_bases, right_bases = peak_prominences(
        y_smooth,
        [global_peak_idx],
        wlen=wlen,
    )
    left_base_idx = int(left_bases[0])
    right_base_idx = int(right_bases[0])

    # Build a straight baseline between the left and right bases.
    x_segment = t[left_base_idx : right_base_idx + 1]
    y_segment = y_smooth[left_base_idx : right_base_idx + 1]
    baseline = np.linspace(
        y_smooth[left_base_idx],
        y_smooth[right_base_idx],
        len(x_segment),
    )

    peak_local_idx = global_peak_idx - left_base_idx
    baseline_at_peak = baseline[peak_local_idx]

    # Main features.
    peak_height = float(y_smooth[global_peak_idx] - baseline_at_peak)
    duration_sec = float((global_peak_idx - left_base_idx) * dt)
    area = float(np.trapz(y_segment - baseline, x_segment))

    # Half-height width: time between 50% height crossings.
    half_level = baseline_at_peak + 0.5 * peak_height
    t_left_half = find_crossing_left(
        t,
        y_smooth,
        half_level,
        left_base_idx,
        global_peak_idx,
    )
    t_right_half = find_crossing_right(
        t,
        y_smooth,
        half_level,
        global_peak_idx,
        right_base_idx,
    )
    width_half = (
        t_right_half - t_left_half
        if np.isfinite(t_left_half) and np.isfinite(t_right_half)
        else np.nan
    )

    # 10% asymmetry: right half at 10% height divided by left half at 10% height.
    p10_level = baseline_at_peak + 0.10 * peak_height
    t_left_10 = find_crossing_left(
        t,
        y_smooth,
        p10_level,
        left_base_idx,
        global_peak_idx,
    )
    t_right_10 = find_crossing_right(
        t,
        y_smooth,
        p10_level,
        global_peak_idx,
        right_base_idx,
    )

    a10 = t[global_peak_idx] - t_left_10 if np.isfinite(t_left_10) else np.nan
    b10 = t_right_10 - t[global_peak_idx] if np.isfinite(t_right_10) else np.nan
    asym_10 = b10 / a10 if np.isfinite(a10) and np.isfinite(b10) and a10 > 0 else np.nan

    return {
        "mode": f"{used_mode}{' + fallback' if fallback_used else ''}",
        "peak_idx": int(global_peak_idx),
        "lb_idx": int(left_base_idx),
        "rb_idx": int(right_base_idx),
        "t_peak_s": float(t[global_peak_idx]),
        "t_lb_s": float(t[left_base_idx]),
        "t_rb_s": float(t[right_base_idx]),
        "peak_height": peak_height,
        "duration_sec": duration_sec,
        "area": area,
        "trim_start_s": float(t[start_idx]),
        "width_half_sec": float(width_half) if np.isfinite(width_half) else np.nan,
        "t_left_half_s": float(t_left_half) if np.isfinite(t_left_half) else np.nan,
        "t_right_half_s": float(t_right_half) if np.isfinite(t_right_half) else np.nan,
        "asym_10": float(asym_10) if np.isfinite(asym_10) else np.nan,
        "snr": float(final_snr) if np.isfinite(final_snr) else np.nan,
    }


# =============================================================================
# Batch processing
# =============================================================================

def extract_fd_number(filename: str) -> Optional[int]:
    """
    Extract the FD concentration/ID number from a filename.

    Example:
        sensor_FD00004_1.npy -> 4
        FD01024_1.npy        -> 1024
    """
    match = re.search(r"FD0*(\d+)_", filename)
    return int(match.group(1)) if match else None


def find_input_files(input_folder: Path, repeat: int) -> List[Path]:
    """
    Find `.npy` files matching the requested repeat number.

    The pattern is intentionally flexible. It accepts names containing
    FDxxxxx_<repeat>.npy, including names like sensor_FD00004_1.npy.
    """
    pattern = re.compile(rf"FD\d{{5}}_{repeat}\.npy$")
    return sorted(path for path in input_folder.glob("*.npy") if pattern.search(path.name))


def process_folder(
    input_folder: Path,
    output_csv: Path,
    repeat: int,
    min_fd: int,
    max_fd: int,
    timestamp_scale: float,
    max_sensor_columns: int,
    butter_order: int,
    cutoff_hz: float,
    min_peak_time_s: float,
    snr_min: float,
) -> pd.DataFrame:
    """
    Process all matching files in a folder and save the feature table.
    """
    input_folder = Path(input_folder)
    output_csv = Path(output_csv)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    files = find_input_files(input_folder, repeat)

    if not files:
        raise FileNotFoundError(
            f"No matching .npy files found in {input_folder} for repeat {repeat}. "
            "Expected filenames ending like FD00004_1.npy."
        )

    rows = []

    for file_path in files:
        fd = extract_fd_number(file_path.name)

        # Skip files outside the requested FD range.
        if fd is None or fd < min_fd or fd > max_fd:
            continue

        try:
            arr = np.load(file_path)
        except Exception as exc:
            rows.append(
                {
                    "file": file_path.name,
                    "FD": fd,
                    "sensor": np.nan,
                    "error": f"Could not load file: {exc}",
                    **empty_feature_result("load_error"),
                }
            )
            continue

        if arr.ndim != 2 or arr.shape[1] < 2:
            rows.append(
                {
                    "file": file_path.name,
                    "FD": fd,
                    "sensor": np.nan,
                    "error": "Array must be 2D with timestamp + at least one sensor column.",
                    **empty_feature_result("invalid_array"),
                }
            )
            continue

        # Sensor columns are 1..N. Column 0 is timestamp.
        last_sensor_col = min(arr.shape[1], max_sensor_columns + 1)

        for col in range(1, last_sensor_col):
            result = process_sensor_channel(
                arr=arr,
                sensor_col=col,
                timestamp_scale=timestamp_scale,
                butter_order=butter_order,
                cutoff_hz=cutoff_hz,
                min_peak_time_s=min_peak_time_s,
                snr_min=snr_min,
            )

            rows.append(
                {
                    "file": file_path.name,
                    "FD": fd,
                    "sensor": f"s{col - 1}",
                    "error": "",
                    **result,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No rows were produced. Check the FD range, repeat number, and input filenames."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return df


# =============================================================================
# Command-line interface
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smooth synchronized sensor .npy files and extract peak features."
    )

    parser.add_argument(
        "--input-folder",
        required=True,
        type=Path,
        help="Folder containing synchronized sensor .npy files from the previous pipeline.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Path where the extracted feature CSV should be saved.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat number to process. Matches files ending with FDxxxxx_<repeat>.npy.",
    )
    parser.add_argument(
        "--min-fd",
        type=int,
        default=4,
        help="Minimum FD value to include.",
    )
    parser.add_argument(
        "--max-fd",
        type=int,
        default=1024,
        help="Maximum FD value to include.",
    )
    parser.add_argument(
        "--timestamp-scale",
        type=float,
        default=1e-9,
        help="Multiplier used to convert timestamp column to seconds. Default assumes nanoseconds.",
    )
    parser.add_argument(
        "--max-sensor-columns",
        type=int,
        default=4,
        help="Maximum number of sensor columns to process after the timestamp column.",
    )
    parser.add_argument(
        "--butter-order",
        type=int,
        default=DEFAULT_BUTTER_ORDER,
        help="Butterworth filter order.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=DEFAULT_CUTOFF_HZ,
        help="Butterworth low-pass cutoff frequency in Hz.",
    )
    parser.add_argument(
        "--min-peak-time-s",
        type=float,
        default=DEFAULT_MIN_PEAK_TIME_S,
        help="Ignore peaks before this time in seconds.",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=DEFAULT_SNR_MIN,
        help="Minimum SNR for relaxed/fallback detections.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = process_folder(
        input_folder=args.input_folder,
        output_csv=args.output_csv,
        repeat=args.repeat,
        min_fd=args.min_fd,
        max_fd=args.max_fd,
        timestamp_scale=args.timestamp_scale,
        max_sensor_columns=args.max_sensor_columns,
        butter_order=args.butter_order,
        cutoff_hz=args.cutoff_hz,
        min_peak_time_s=args.min_peak_time_s,
        snr_min=args.snr_min,
    )

    print(f"Processed rows: {len(df)}")
    print(f"Saved features to: {args.output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()
