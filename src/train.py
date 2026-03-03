import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_data, clean_data, encode_features, split_data


def train_model():
    # Robust path handling (production style)
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.xlsx"

    # Load
    df = load_data(DATA_PATH)

    # Clean
    df = clean_data(df)

    # Encode
    df_encoded, encoders = encode_features(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df_encoded)

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model and encoders
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model, model_dir / "churn_model.pkl")
    joblib.dump(encoders, model_dir / "label_encoders.pkl")

    print("✅ Model training complete. Saved to models/")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()