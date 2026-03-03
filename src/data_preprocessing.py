import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from Excel file.
    """
    df = pd.read_excel(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Strip spaces from column names
    - Drop unnecessary columns
    - Convert Total Charges to numeric
    - Handle missing values
    """
    df = df.copy()

    # Remove leading/trailing spaces in column names
    df.columns = df.columns.str.strip()

    # Columns not useful for prediction
    drop_cols = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "CLTV",
        "Churn Reason"
    ]

    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Convert Total Charges to numeric
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"].fillna(df["Total Charges"].median(), inplace=True)

    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical features using Label Encoding.
    """
    df = df.copy()
    label_encoders = {}

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


def split_data(df: pd.DataFrame, target: str = "Churn Value"):
    """
    Split dataset into train and test sets.
    """
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)