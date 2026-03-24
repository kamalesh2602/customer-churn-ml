import joblib
import pandas as pd
from app import CustomerData

pipeline = joblib.load(r'c:\Users\ELCOT\Desktop\kamalesh\customer-churn-ml\models\churn_pipeline.pkl')
print("Pipeline feature names:", pipeline.feature_names_in_)

# Dummy data
data = CustomerData(
    Gender="Male",
    Senior_Citizen=0,
    Partner="Yes",
    Dependents="No",
    Tenure_Months=1,
    Phone_Service="Yes",
    Multiple_Lines="No",
    Internet_Service="DSL",
    Online_Security="No",
    Online_Backup="No",
    Device_Protection="No",
    Tech_Support="No",
    Streaming_TV="No",
    Streaming_Movies="No",
    Contract="Month-to-month",
    Paperless_Billing="Yes",
    Payment_Method="Electronic check",
    Monthly_Charges=10.0,
    Total_Charges=10.0
)

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

for new_key, old_key in column_mapping.items():
    input_dict[old_key] = input_dict.pop(new_key)
    
df = pd.DataFrame([input_dict])
print("DF columns before sorting:", df.columns.tolist())

numeric_cols = [
    "Senior Citizen",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
categorical_cols = [col for col in df.columns if col not in numeric_cols]
for col in categorical_cols:
    df[col] = df[col].astype(str)

print("DF columns right before prediction:", df.columns.tolist())
df = df[pipeline.feature_names_in_]
print("DF columns after sorting:", df.columns.tolist())

# Check data types
print("DF dtypes:\n", df.dtypes)

try:
    prediction = pipeline.predict(df)[0]
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
