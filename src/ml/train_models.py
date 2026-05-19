import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.ml.dataset import load_feature_dataset
from src.ml.model_factory import build_model


def train_one_model(args):
    X, y, data = load_feature_dataset(
        feature_dir=args.feature_dir,
        map_csv=args.map_csv,
        target_column=args.target,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(args.model, random_state=args.random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{args.model}_{args.target}.joblib"
    report_path = output_dir / f"{args.model}_{args.target}_report.txt"
    cm_path = output_dir / f"{args.model}_{args.target}_confusion_matrix.csv"

    joblib.dump(model, model_path)

    with open(report_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    pd.DataFrame(cm).to_csv(cm_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Accuracy: {accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--map-csv", required=True)
    parser.add_argument("--model", choices=["rf", "xgboost", "voting"], required=True)
    parser.add_argument("--target", default="functional_group")
    parser.add_argument("--output-dir", default="outputs/models")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_model(args)