from pathlib import Path
import pandas as pd


def load_feature_dataset(feature_dir, map_csv, target_column="functional_group"):
    """
    Load multiple feature CSV files and attach labels from compound_map.csv.

    Parameters
    ----------
    feature_dir:
        Folder containing feature CSV files.
    map_csv:
        CSV file with columns:
        filename, compound, functional_group, class_label
    target_column:
        Which column to predict, for example:
        'compound', 'functional_group', or 'class_label'.
    """

    feature_dir = Path(feature_dir)
    compound_map = pd.read_csv(map_csv)

    all_frames = []

    for _, row in compound_map.iterrows():
        feature_file = feature_dir / row["filename"]

        if not feature_file.exists():
            print(f"Skipping missing file: {feature_file}")
            continue

        df = pd.read_csv(feature_file)

        df["compound"] = row["compound"]
        df["functional_group"] = row["functional_group"]
        df["class_label"] = row["class_label"]

        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No feature CSV files were loaded.")

    data = pd.concat(all_frames, ignore_index=True)

    drop_cols = [
        "file",
        "FD",
        "sensor",
        "mode",
        "compound",
        "functional_group",
        "class_label",
    ]

    feature_cols = [
        col for col in data.columns
        if col not in drop_cols
    ]

    X = data[feature_cols].copy()
    y = data[target_column].copy()

    # Replace invalid values before training.
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, data