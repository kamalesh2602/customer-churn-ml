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

    # Rename keys back to original column format
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

    for new_key, old_key in column_mapping.items():
        input_dict[old_key] = input_dict.pop(new_key)

    df = pd.DataFrame([input_dict])

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    }