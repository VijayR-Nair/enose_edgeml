#!/usr/bin/env python3
"""
Single-file GC + sensor preprocessing pipeline.

This script does NOT include feature extraction / peak metrics.

It can:
1. Combine multiple .h5 sensor files into one sorted .npy file.
2. Convert GC .txt files into GC .npy files with absolute timestamps.
3. Synchronize/slice the combined sensor data for each GC file.

Example:
    python gc_sensor_pipeline.py \
        --sensor-h5-dir "/path/to/Sensor_Data" \
        --combined-sensor-npy "/path/to/Sensor_Data/combined_sensor.npy" \
        --gc-txt-dir "/path/to/GC_Data" \
        --gc-npy-dir "/path/to/GC_NPY" \
        --synced-output-dir "/path/to/Synced_Sensor"

Interactive mode:
    python gc_sensor_pipeline.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------
# Default paths from your notebook
# ---------------------------------------------------------------------

DEFAULT_BASE_DIR = Path(
    "/home/vijayrnair/IVV_JSON/Data Pro/Raw_Data/"
    "GC_SOMSA_sensitivity_decanoic_acid_split"
)

DEFAULT_SENSOR_H5_DIR = DEFAULT_BASE_DIR / "Sensor_Data"
DEFAULT_COMBINED_SENSOR_NPY = DEFAULT_SENSOR_H5_DIR / "comb_decanooic.npy"
DEFAULT_GC_TXT_DIR = DEFAULT_BASE_DIR / "GC_Data"
DEFAULT_GC_NPY_DIR = DEFAULT_GC_TXT_DIR
DEFAULT_SYNCED_OUTPUT_DIR = DEFAULT_BASE_DIR / "Synced_Sensor"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def find_file_type(directory: str | Path, ending: str) -> list[Path]:
    """Return all files under directory ending with the given suffix."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob(f"*{ending}") if path.is_file())


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def ask_path(prompt: str, default: Path | None = None) -> Path:
    """Interactive path input with optional default."""
    if default is not None:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return Path(user_input) if user_input else default
    return Path(input(f"{prompt}: ").strip())


# ---------------------------------------------------------------------
# Step 1: Combine H5 sensor files
# ---------------------------------------------------------------------

def load_single_sensor_h5(
    h5_file_path: str | Path,
    sensor_channel_path: str = "channel1/sgp40",
    invert_raw_values: bool = True,
) -> np.ndarray:
    """
    Load one H5 sensor file and return columns:
    [timestamp, sensor_0, sensor_1, sensor_2, sensor_3]
    """
    h5_file_path = Path(h5_file_path)

    with h5py.File(h5_file_path, "r") as f:
        if "measurement" not in f:
            raise ValueError("No 'measurement' group found.")

        measurement_keys = list(f["measurement"].keys())
        if not measurement_keys:
            raise ValueError("No measurement key found.")

        measurement_key = measurement_keys[0]
        base_path = f"measurement/{measurement_key}/{sensor_channel_path}"

        if base_path not in f:
            raise ValueError(f"Sensor path not found: {base_path}")

        base = f[base_path]

        timestamps = base["timestamp"][:, 0, 0]
        s0 = base["sensor_0/sensor_raw"][:, 0, 0]
        s1 = base["sensor_1/sensor_raw"][:, 0, 0]
        s2 = base["sensor_2/sensor_raw"][:, 0, 0]
        s3 = base["sensor_3/sensor_raw"][:, 0, 0]

        if invert_raw_values:
            s0 = 65535 - s0
            s1 = 65535 - s1
            s2 = 65535 - s2
            s3 = 65535 - s3

        return np.column_stack((timestamps, s0, s1, s2, s3))


def combine_sensor_h5_files(
    input_dir: str | Path,
    output_npy_path: str | Path,
    sensor_channel_path: str = "channel1/sgp40",
    invert_raw_values: bool = True,
) -> np.ndarray | None:
    """
    Combine all .h5 files in input_dir into one sorted .npy file.
    """
    input_dir = Path(input_dir)
    output_npy_path = Path(output_npy_path)

    h5_files = find_file_type(input_dir, ".h5")
    if not h5_files:
        print(f"No .h5 files found in: {input_dir}")
        return None

    all_data = []

    for h5_path in h5_files:
        try:
            data = load_single_sensor_h5(
                h5_path,
                sensor_channel_path=sensor_channel_path,
                invert_raw_values=invert_raw_values,
            )
            all_data.append(data)
            print(f"Loaded {h5_path.name}: shape {data.shape}")
        except Exception as exc:
            print(f"Skipping {h5_path.name}: {exc}")

    if not all_data:
        print("No valid sensor data found to save.")
        return None

    combined_data = np.vstack(all_data)
    combined_data = combined_data[combined_data[:, 0].argsort()]

    ensure_dir(output_npy_path.parent)
    np.save(output_npy_path, combined_data)

    print(f"Saved combined sensor data: {output_npy_path}")
    print(f"Combined sensor shape: {combined_data.shape}")

    return combined_data


# ---------------------------------------------------------------------
# Step 2: Convert GC TXT files to NPY
# ---------------------------------------------------------------------

# Matches:
#   8/6/2025 6:04:32 AM
#   8/6/2025 6:04:32 AM +02:00
#   8/6/2025 6:04:32 AM +0200
INJECT_RE = re.compile(
    r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\s+"
    r"(?P<hour12>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})\s*"
    r"(?P<ampm>AM|PM)\s*"
    r"(?P<tz>[+-]\d{2}:?\d{2})?",
    flags=re.IGNORECASE,
)


def parse_inject_time_us(line: str) -> float:
    """
    Extract Inject Time from a line and return Unix time in microseconds.
    If no timezone is provided, UTC is assumed.
    """
    match = INJECT_RE.search(line)
    if not match:
        raise ValueError(f"Could not find timestamp in Inject Time line: {line!r}")

    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour12 = int(match.group("hour12"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    ampm = match.group("ampm").upper()
    tz_s = match.group("tz")

    if ampm == "AM":
        hour24 = 0 if hour12 == 12 else hour12
    else:
        hour24 = 12 if hour12 == 12 else hour12 + 12

    dt = _dt.datetime(year, month, day, hour24, minute, second)

    if tz_s:
        tz_clean = tz_s.replace(":", "")
        sign = 1 if tz_clean[0] == "+" else -1
        hours = int(tz_clean[1:3])
        minutes = int(tz_clean[3:5])
        offset = _dt.timedelta(hours=hours, minutes=minutes) * sign
        tzinfo = _dt.timezone(offset)
    else:
        tzinfo = _dt.timezone.utc

    dt = dt.replace(tzinfo=tzinfo)
    return dt.timestamp() * 1_000_000


def convert_single_gc_txt_to_npy(
    txt_path: str | Path,
    output_dir: str | Path,
    skip_existing: bool = True,
) -> Path | None:
    """
    Convert one GC .txt file into .npy with columns:
    [absolute_timestamp_us, value]
    """
    txt_path = Path(txt_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    output_path = output_dir / f"{txt_path.stem}.npy"

    if skip_existing and output_path.exists():
        print(f"GC measurement already converted: {output_path.name}. Skipping.")
        return output_path

    with open(txt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        all_lines = f.readlines()

    injection_time_us = None

    for line in all_lines:
        if "Inject Time" in line:
            injection_time_us = parse_inject_time_us(line)
            break

    if injection_time_us is None:
        raise ValueError(f"No 'Inject Time' line found in file: {txt_path}")

    gc_rows = []
    is_data_section = False
    line_index = 0

    while line_index < len(all_lines):
        line = all_lines[line_index]

        if not is_data_section and ("Time (min)" in line and "Value (counts)" in line):
            is_data_section = True
            line_index += 1
            continue

        if not is_data_section:
            line_index += 1
            continue

        parts = [part.replace(",", "") for part in line.split()]

        if not parts:
            line_index += 1
            continue

        try:
            time_min = float(parts[0])

            # Common format:
            # Time Step Value
            if len(parts) >= 3:
                value = float(parts[2])
                line_index += 1

            # Fallback format:
            # time is on one line, value appears on next line
            else:
                line_index += 1
                value_line = all_lines[line_index]
                value_parts = [part.replace(",", "") for part in value_line.split()]
                value = float(value_parts[-1])
                line_index += 1

            absolute_timestamp_us = injection_time_us + time_min * 60_000_000
            gc_rows.append([absolute_timestamp_us, value])

        except (ValueError, IndexError):
            line_index += 1
            continue

    gc_data = np.asarray(gc_rows, dtype=float)

    if gc_data.size == 0:
        print(f"Warning: no numeric GC data parsed in {txt_path.name}. Saving empty array.")

    np.save(output_path, gc_data)
    print(f"Saved GC npy: {output_path} shape={gc_data.shape}")

    return output_path


def convert_all_gc_txt_files(
    gc_txt_dir: str | Path,
    output_dir: str | Path,
    skip_existing: bool = True,
) -> list[Path]:
    """Convert all GC .txt files in a directory tree to .npy files."""
    txt_files = find_file_type(gc_txt_dir, ".txt")

    if not txt_files:
        print(f"No GC .txt files found in: {gc_txt_dir}")
        return []

    output_paths = []

    for txt_path in txt_files:
        try:
            output_path = convert_single_gc_txt_to_npy(
                txt_path,
                output_dir=output_dir,
                skip_existing=skip_existing,
            )
            if output_path is not None:
                output_paths.append(output_path)
        except Exception as exc:
            print(f"Skipping {txt_path.name}: {exc}")

    return output_paths


# ---------------------------------------------------------------------
# Step 3: Synchronize sensor data with GC files
# ---------------------------------------------------------------------

def sync_sensor_with_gc(
    gc_npy_dir: str | Path,
    sensor_npy_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    For each GC .npy file, slice the sensor data between the GC start/end
    timestamps and save as sensor_<gc_filename>.npy.
    """
    gc_npy_dir = Path(gc_npy_dir)
    sensor_npy_path = Path(sensor_npy_path)
    output_dir = Path(output_dir)

    if not sensor_npy_path.exists():
        raise FileNotFoundError(f"Sensor .npy file not found: {sensor_npy_path}")

    ensure_dir(output_dir)

    sensor_data = np.load(sensor_npy_path)
    print(f"Sensor data loaded: {sensor_data.shape}")

    if sensor_data.ndim != 2 or sensor_data.shape[1] < 2:
        raise ValueError("Sensor data must be a 2D array with timestamp in column 0.")

    gc_npy_files = find_file_type(gc_npy_dir, ".npy")
    print(f"GC .npy files found: {len(gc_npy_files)}")

    saved_paths = []

    for gc_path in gc_npy_files:
        try:
            gc_data = np.load(gc_path)

            if gc_data.size == 0:
                print(f"Skipping {gc_path.name}: GC data is empty.")
                continue

            if gc_data.ndim != 2 or gc_data.shape[1] < 2:
                print(f"Skipping {gc_path.name}: expected 2D GC array with at least 2 columns.")
                continue

            gc_filename = gc_path.stem

            start_time = gc_data[0, 0]
            finish_time = gc_data[-1, 0]

            index_start = int(np.argmin(np.abs(sensor_data[:, 0] - start_time)))
            index_finish = int(np.argmin(np.abs(sensor_data[:, 0] - finish_time)))

            if index_finish <= index_start:
                print(f"Skipping {gc_path.name}: invalid sensor overlap window.")
                continue

            sensor_segment = sensor_data[index_start:index_finish]

            output_path = output_dir / f"sensor_{gc_filename}.npy"
            np.save(output_path, sensor_segment)
            saved_paths.append(output_path)

            print(f"Saved sensor segment for {gc_filename}: {output_path} shape={sensor_segment.shape}")

        except Exception as exc:
            print(f"Skipping {gc_path.name}: {exc}")

    return saved_paths


# ---------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Run selected pipeline steps."""
    if args.combine_sensor:
        print("\nStep 1: Combining sensor H5 files")
        combine_sensor_h5_files(
            input_dir=args.sensor_h5_dir,
            output_npy_path=args.combined_sensor_npy,
            sensor_channel_path=args.sensor_channel_path,
            invert_raw_values=not args.no_invert_raw,
        )

    if args.convert_gc:
        print("\nStep 2: Converting GC TXT files to NPY")
        convert_all_gc_txt_files(
            gc_txt_dir=args.gc_txt_dir,
            output_dir=args.gc_npy_dir,
            skip_existing=not args.overwrite_gc_npy,
        )

    if args.sync:
        print("\nStep 3: Synchronizing sensor data with GC")
        sync_sensor_with_gc(
            gc_npy_dir=args.gc_npy_dir,
            sensor_npy_path=args.combined_sensor_npy,
            output_dir=args.synced_output_dir,
        )

    print("\nDone.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess GC and sensor files. Feature extraction is not included."
    )

    parser.add_argument("--sensor-h5-dir", type=Path, default=DEFAULT_SENSOR_H5_DIR)
    parser.add_argument("--combined-sensor-npy", type=Path, default=DEFAULT_COMBINED_SENSOR_NPY)
    parser.add_argument("--gc-txt-dir", type=Path, default=DEFAULT_GC_TXT_DIR)
    parser.add_argument("--gc-npy-dir", type=Path, default=DEFAULT_GC_NPY_DIR)
    parser.add_argument("--synced-output-dir", type=Path, default=DEFAULT_SYNCED_OUTPUT_DIR)

    parser.add_argument(
        "--sensor-channel-path",
        default="channel1/sgp40",
        help="H5 path after measurement/<key>/, for example channel1/sgp40 or channel0/sgp40.",
    )

    parser.add_argument(
        "--no-invert-raw",
        action="store_true",
        help="Disable 65535 - raw sensor conversion.",
    )

    parser.add_argument(
        "--overwrite-gc-npy",
        action="store_true",
        help="Recreate GC .npy files even if they already exist.",
    )

    parser.add_argument(
        "--combine-sensor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run H5 sensor combination step.",
    )
    parser.add_argument(
        "--convert-gc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run GC TXT-to-NPY conversion step.",
    )
    parser.add_argument(
        "--sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run sensor/GC synchronization step.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for paths interactively instead of using defaults/CLI args.",
    )

    return parser


def fill_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    print("\nEnter paths. Press Enter to keep the shown default.")

    args.sensor_h5_dir = ask_path("Sensor H5 folder", args.sensor_h5_dir)
    args.combined_sensor_npy = ask_path("Output combined sensor .npy path", args.combined_sensor_npy)
    args.gc_txt_dir = ask_path("GC TXT folder", args.gc_txt_dir)
    args.gc_npy_dir = ask_path("Output GC NPY folder", args.gc_npy_dir)
    args.synced_output_dir = ask_path("Output synchronized sensor folder", args.synced_output_dir)

    channel = input(f"Sensor channel path [{args.sensor_channel_path}]: ").strip()
    if channel:
        args.sensor_channel_path = channel

    return args


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.interactive:
        args = fill_interactive_args(args)

    run_pipeline(args)


if __name__ == "__main__":
    main()
