
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/Career_Path_Dataset_with_Career.csv")

print("Original Dataset Shape:", df.shape)


# ==========================================
# 2. DATA CLEANING
# ==========================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df["Certifications"] = df["Certifications"].fillna("Not Provided")
df["Past_Jobs_Internships"] = df["Past_Jobs_Internships"].fillna("Not Provided")
df["Achievements"] = df["Achievements"].fillna("Not Provided")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("dataset/cleaned_career_dataset.csv", index=False)

print("\nCleaned Dataset Shape:", df.shape)
print("Cleaned dataset saved successfully!")


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

# Remove Skills_Gained to avoid possible data leakage

X = df.drop(
    columns=["Suggested_Career_Path", "Skills_Gained"]
)

y = df["Suggested_Career_Path"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
print("Total Career Categories:", y.nunique())


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

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


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

random_forest_model = Pipeline(steps=[
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

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, random_forest_pred))

print("\nClassification Report:")
print(classification_report(y_test, random_forest_pred, zero_division=0))

joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

print("Random Forest model saved successfully!")


# ==========================================
# 7. LOGISTIC REGRESSION MODEL
# ==========================================

logistic_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

logistic_model = Pipeline(steps=[
    ("preprocessor", logistic_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, logistic_pred))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred, zero_division=0))

joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)

print("Logistic Regression model saved successfully!")


# ==========================================
# 8. DECISION TREE MODEL
# ==========================================

decision_tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    ))
])

decision_tree_model.fit(X_train, y_train)

decision_tree_pred = decision_tree_model.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, decision_tree_pred))

print("\nClassification Report:")
print(classification_report(y_test, decision_tree_pred, zero_division=0))

joblib.dump(
    decision_tree_model,
    "models/decision_tree_model.pkl"
)

print("Decision Tree model saved successfully!")


# ==========================================
# 9. CONFUSION MATRICES
# ==========================================

models = [
    ("Random Forest", random_forest_model),
    ("Logistic Regression", logistic_model),
    ("Decision Tree", decision_tree_model)
]

for model_name, trained_model in models:

    print(f"\nGenerating {model_name} Confusion Matrix")

    ConfusionMatrixDisplay.from_estimator(
        trained_model,
        X_test,
        y_test,
        xticks_rotation="vertical"
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    file_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/{file_name}_confusion_matrix.png"
    )

    plt.close()

print("\nAll Confusion Matrices saved successfully!")


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

predictions = {
    "Random Forest": random_forest_pred,
    "Logistic Regression": logistic_pred,
    "Decision Tree": decision_tree_pred
}

results = []

for model_name, predictions_values in predictions.items():

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions_values),
        "Precision": precision_score(
            y_test, predictions_values, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions_values, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions_values, average="weighted", zero_division=0
        )
    })

comparison = pd.DataFrame(results)

print("\nFinal Model Comparison:")
print(comparison.round(3))

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# ==========================================
# 11. DATASET CATEGORY INFORMATION
# ==========================================

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


print("\n================================")
print("ALL THREE MODELS TRAINED SUCCESSFULLY!")
print("================================")
