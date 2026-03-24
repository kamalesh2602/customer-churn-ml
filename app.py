from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Customer Churn Prediction API")

# Load trained pipeline
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)


# Define input schema
class CustomerData(BaseModel):
    Gender: str
    Senior_Citizen: int
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()

    column_mapping = {
        "Senior_Citizen": "Senior Citizen",
        "Tenure_Months": "Tenure Months",
        "Phone_Service": "Phone Service",
        "Multiple_Lines": "Multiple Lines",
        "Internet_Service": "Internet Service",
        "Online_Security": "Online Security",
        "Online_Backup": "Online Backup",
        "Device_Protection": "Device Protection",
        "Tech_Support": "Tech Support",
        "Streaming_TV": "Streaming TV",
        "Streaming_Movies": "Streaming Movies",
        "Paperless_Billing": "Paperless Billing",
        "Payment_Method": "Payment Method",
        "Monthly_Charges": "Monthly Charges",
        "Total_Charges": "Total Charges"
    }

    # Rename keys
    for new_key, old_key in column_mapping.items():
        input_dict[old_key] = input_dict.pop(new_key)

    df = pd.DataFrame([input_dict])

    # 🔥 FORCE TYPES EXACTLY LIKE TRAINING
    
    # Map Senior Citizen from int (0/1) to Yes/No to match training data
    if "Senior Citizen" in df.columns:
        df["Senior Citizen"] = df["Senior Citizen"].map({1: "Yes", 0: "No", "1": "Yes", "0": "No"}).fillna(df["Senior Citizen"])

    # Numeric columns
    numeric_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical columns → force string
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # 🔥 Ensure same column order
    df = df[pipeline.feature_names_in_]

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    }
    