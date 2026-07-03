"""
Run batch predictions on a CSV of new customers.

Usage:
    python src/predict.py --input data/new_customers.csv --output data/predictions.csv
"""
import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_data, encode_features  # noqa: E402

MODEL_PATH = "models/churn_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    saved = joblib.load(ENCODERS_PATH)
    encoders, columns = saved["encoders"], saved["columns"]

    df = pd.read_csv(args.input)
    original = df.copy()

    df = clean_data(df) if "Churn" not in df.columns else clean_data(df).drop(columns=["Churn"])
    df, _ = encode_features(df, encoders=encoders)
    df = df.reindex(columns=columns, fill_value=0)

    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    original["churn_prediction"] = preds
    original["churn_probability"] = probs.round(3)
    original.to_csv(args.output, index=False)
    print(f"Wrote {len(original)} predictions to {args.output}")


if __name__ == "__main__":
    main()
