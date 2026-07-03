"""
Data loading and preprocessing for the Telco Customer Churn dataset.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_data(local_path="data/Telco-Customer-Churn.csv"):
    """Load the dataset, downloading it if not already present locally."""
    import os

    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(DATA_URL)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix types, drop IDs, handle missing values."""
    df = df.copy()

    # TotalCharges has some blank strings for new customers; coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame, encoders: dict = None):
    """
    Label-encode categorical columns. If `encoders` is provided (from a
    previous training run), reuse them for consistent transformation on
    new/inference data. Otherwise fit new encoders and return them.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    fit_new = encoders is None
    if fit_new:
        encoders = {}

    for col in categorical_cols:
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # map unseen categories to the first known class to avoid crashes
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def prepare_dataset(local_path="data/Telco-Customer-Churn.csv"):
    """Full pipeline: load -> clean -> encode. Returns X, y, encoders."""
    df = load_data(local_path)
    df = clean_data(df)

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    X_encoded, encoders = encode_features(X)
    return X_encoded, y, encoders
