import joblib
import pandas as pd

pipeline = joblib.load(r'c:\Users\ELCOT\Desktop\kamalesh\customer-churn-ml\models\churn_pipeline.pkl')
features = list(pipeline.feature_names_in_)

with open('features.txt', 'w') as f:
    f.write('\n'.join(features))

print("Features written to features.txt")
