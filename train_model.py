import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load original dataset
df = pd.read_csv("dataset/Career_Path_Dataset_with_Career.csv")

print("Original Dataset Shape:", df.shape)

# Check missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values in text columns
df["Certifications"] = df["Certifications"].fillna("Not Provided")
df["Past_Jobs_Internships"] = df["Past_Jobs_Internships"].fillna("Not Provided")
df["Achievements"] = df["Achievements"].fillna("Not Provided")

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("dataset/cleaned_career_dataset.csv", index=False)

print("\nCleaned Dataset Shape:", df.shape)
print("\nCleaned dataset saved successfully!")
# Separate Features and Target

X = df.drop("Suggested_Career_Path", axis=1)

y = df["Suggested_Career_Path"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Classes:")
print(y.nunique())
# Inspect important features

print("\nSample Input Features:")
print(X[[
    "Technical_Skills",
    "Fields_of_Interest",
    "Skills_Gained"
]].head(5))

print("\nSample Target Values:")
print(y.head(5))
from sklearn.model_selection import train_test_split

# Remove Skills_Gained to avoid possible data leakage
X = df.drop(
    columns=["Suggested_Career_Path", "Skills_Gained"]
)

y = df["Suggested_Career_Path"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Targets:", y_train.shape)
print("Testing Targets:", y_test.shape)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["str", "object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=1
))
])

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nRandom Forest Model Results")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib

# Save trained Random Forest model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Preprocessing for Logistic Regression
logistic_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Create Logistic Regression model
logistic_model = Pipeline(steps=[
    ("preprocessor", logistic_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

# Train model
logistic_model.fit(X_train, y_train)

# Predictions
logistic_pred = logistic_model.predict(X_test)

# Evaluation
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, logistic_pred))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred))

# Save model
joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)

print("\nLogistic Regression model saved successfully!")
# Confusion Matrix for Random Forest
print("\nGenerating Random Forest Confusion Matrix")

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    xticks_rotation="vertical",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/random_forest_confusion_matrix.png")
plt.show()


# Confusion Matrix for Logistic Regression
print("\nGenerating Logistic Regression Confusion Matrix")

ConfusionMatrixDisplay.from_estimator(
    logistic_model,
    X_test,
    y_test,
    xticks_rotation="vertical",
    cmap="Greens"
)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/logistic_regression_confusion_matrix.png")
plt.show()

print("\nBoth Confusion Matrices saved successfully!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model Comparison
comparison = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, logistic_pred)
    ],
    "Precision": [
        precision_score(y_test, y_pred, average="weighted"),
        precision_score(y_test, logistic_pred, average="weighted")
    ],
    "Recall": [
        recall_score(y_test, y_pred, average="weighted"),
        recall_score(y_test, logistic_pred, average="weighted")
    ],
    "F1 Score": [
        f1_score(y_test, y_pred, average="weighted"),
        f1_score(y_test, logistic_pred, average="weighted")
    ]
})

print("\nModel Comparison:")
print(comparison.round(3))

comparison.to_csv("outputs/model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")
print("\nACTUAL DATASET CATEGORIES")

for column in [
    "Gender",
    "Highest_qualification",
    "Stream",
    "Current_Academic_Level",
    "Preferred_Work_Style",
    "Work_Type_Interest",
    "Willing_to_Relocate"
]:
    print(f"\n{column}:")
    print(df[column].unique())