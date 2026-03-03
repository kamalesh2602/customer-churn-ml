import joblib
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import load_data, clean_data


def train_model():
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.xlsx"

    # Load & clean
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Define target
    target = "Churn Value"
    X = df.drop(columns=[target])
    y = df[target]

    # Separate column types
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    # Save pipeline
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(pipeline, model_dir / "churn_pipeline.pkl")

    print("✅ Pipeline training complete. Saved to models/")

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train_model()