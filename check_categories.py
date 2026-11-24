import pandas as pd

df = pd.read_csv("dataset/cleaned_career_dataset.csv")

print("ACTUAL DATASET CATEGORIES")

columns = [
    "Gender",
    "Highest_qualification",
    "Stream",
    "Current_Academic_Level",
    "Preferred_Work_Style",
    "Work_Type_Interest",
    "Willing_to_Relocate"
]

for column in columns:
    print(f"\n{column}:")
    print(df[column].unique())