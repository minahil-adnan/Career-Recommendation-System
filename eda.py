import pandas as pd

df = pd.read_csv("dataset/Career_Path_Dataset_with_Career.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())
# Dataset Audit Summary

print("\nDataset Audit Summary")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("Total Missing Values:", df.isnull().sum().sum())

print("Total Duplicate Rows:", df.duplicated().sum())

print("\nCareer Categories:")
print(df["Suggested_Career_Path"].nunique())

print("\nCareer Distribution:")
print(df["Suggested_Career_Path"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Career Distribution
plt.figure(figsize=(14, 6))
sns.countplot(data=df, y="Suggested_Career_Path",
              order=df["Suggested_Career_Path"].value_counts().index)
plt.title("Career Category Distribution")
plt.xlabel("Number of Students")
plt.ylabel("Career")
plt.tight_layout()
plt.savefig("outputs/career_distribution.png")
plt.show()

# 2. Missing Values
plt.figure(figsize=(10, 5))
df.isnull().sum().plot(kind="bar")
plt.title("Missing Values in Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/missing_values.png")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=15, kde=True)
plt.title("Student Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png")
plt.show()

# 4. CGPA Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Grade_CGPA_Percentage"], bins=15, kde=True)
plt.title("CGPA / Percentage Distribution")
plt.xlabel("Grade / CGPA / Percentage")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/grade_distribution.png")
plt.show()

print("\nAll 4 visualizations saved in outputs folder.")