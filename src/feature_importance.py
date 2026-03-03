import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_feature_importance():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "models" / "churn_pipeline.pkl"

    pipeline = joblib.load(model_path)

    # Extract trained model
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get feature names
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )

    feature_names = list(numeric_features) + list(categorical_features)

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot top 20
    top_features = feature_importance_df.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(BASE_DIR / "models" / "feature_importance.png")
    plt.show()

    print("✅ Feature importance plot saved to models/")


if __name__ == "__main__":
    plot_feature_importance()