import joblib

pipeline = joblib.load(r'c:\Users\ELCOT\Desktop\kamalesh\customer-churn-ml\models\churn_pipeline.pkl')
preprocessor = pipeline.named_steps['preprocessor']
for name, transformer, cols in preprocessor.transformers_:
    print(f"{name}: {cols}")
