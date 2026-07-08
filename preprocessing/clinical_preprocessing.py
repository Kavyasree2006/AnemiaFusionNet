import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset/full_dataset.csv")

# ENCODE CATEGORICALS
categorical_cols = [
    "gender",
    "fatigue",
    "diet",
    "region"
]

encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# NUMERICAL FEATURES
num_cols = [
    "age",
    "hb",
    "bmi",
    "geo_risk"
]

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

# SAVE
df.to_csv("dataset/processed_clinical.csv", index=False)

print("Clinical preprocessing completed")