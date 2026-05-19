import argparse
from pathlib import Path

import joblib
import pandas as pd


DROP_COLS = [
    "file",
    "FD",
    "sensor",
    "mode",
    "compound",
    "functional_group",
    "class_label",
]


def predict(args):
    model = joblib.load(args.model_path)
    df = pd.read_csv(args.input_csv)

    feature_cols = [
        col for col in df.columns
        if col not in DROP_COLS
    ]

    X = df[feature_cols].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median(numeric_only=True))

    predictions = model.predict(X)

    df["prediction"] = predictions

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved predictions: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)