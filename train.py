"""
Train a churn prediction model and save the artifacts.

Usage:
    python src/train.py
"""
import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from preprocess import prepare_dataset  # noqa: E402

MODEL_PATH = "models/churn_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"


def main():
    print("Loading and preparing data...")
    X, y, encoders = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Feature importance
    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: -x[1]
    )
    print("Top 5 features:")
    for name, score in importances[:5]:
        print(f"  {name}: {score:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"encoders": encoders, "columns": list(X.columns)}, ENCODERS_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved encoders to {ENCODERS_PATH}")


if __name__ == "__main__":
    main()
