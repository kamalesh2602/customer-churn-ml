import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import load_data, clean_data


def train_model():
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.xlsx"

    # Load & clean
    df = load_data(DATA_PATH)
    df = clean_data(df)

    target = "Churn Value"
    X = df.drop(columns=[target])
    y = df[target]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    # Hyperparameter grid
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(best_model, model_dir / "churn_pipeline.pkl")

    print("✅ Best tuned pipeline saved.")

    return best_model, X_test, y_test


if __name__ == "__main__":
    train_model()