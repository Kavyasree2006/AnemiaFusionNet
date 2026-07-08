import pandas as pd

df = pd.read_excel("dataset/anemia_dataset_completed.xlsx")

df.to_csv("dataset/full_dataset.csv", index=False)

print("CSV created successfully")