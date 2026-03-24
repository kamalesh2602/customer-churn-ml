import pandas as pd
df = pd.read_excel('c:/Users/ELCOT/Desktop/kamalesh/customer-churn-ml/data/telco_churn.xlsx')
with open('dtypes.txt', 'w') as f:
    f.write(str(df.dtypes))
