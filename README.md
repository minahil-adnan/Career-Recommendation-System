# Career Recommendation System

## Project Overview

The Career Recommendation System is a Machine Learning project that predicts a suitable career path based on a person's educational background, technical skills, interests, and work preferences.

The system uses supervised machine learning classification models to recommend a career category.

## Objectives

- Analyze career-related data.
- Clean and preprocess the dataset.
- Explore data through visualizations.
- Train and compare machine learning models.
- Perform hyperparameter experiments.
- Build an interactive Streamlit application.

## Dataset

- **Dataset Name:** Career Path Dataset with Career
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/cns7h4b9g7/1)
- **Total Records:** 15,000
- **Total Columns:** 19
- **Target Column:** `Suggested_Career_Path`
- **Career Categories:** 22

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Pytest

## Data Preprocessing

- Checked missing values.
- Handled missing values in selected columns using "Not Provided".
- Checked duplicate records.
- Separated features and target.
- Excluded `Skills_Gained` from model features to reduce the risk of target leakage.
- Applied One-Hot Encoding to categorical features.
- Split the dataset into training and testing sets using an 80:20 ratio.

## Exploratory Data Analysis (EDA)

The following visualizations were created:

1. Career Distribution
2. Missing Values
3. Age Distribution
4. Grade Distribution

The visualizations are saved in the `outputs` folder.

## Machine Learning Models

Three supervised classification models were implemented:

1. Random Forest Classifier
2. Logistic Regression
3. Decision Tree Classifier

The models were evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-Score
- Confusion Matrix

## Model Results

The following results were obtained on the test dataset:

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Random Forest | 31.33% | 31.30% |
| Logistic Regression | 33.53% | 33.50% |
| Decision Tree | 33.47% | 27.13% |

Logistic Regression achieved the highest accuracy and weighted F1-score among the three base models.

The detailed comparison is available in:

`outputs/model_comparison.csv`

## Hyperparameter Experiments

Two additional experiments were performed to compare different model settings.

### Experiment 1: Random Forest

Different numbers of trees and maximum-depth settings were tested.

Results are saved in:

`outputs/random_forest_experiments.csv`

### Experiment 2: Decision Tree

Different maximum-depth settings were tested:

- Depth 5
- Depth 10
- Depth 15

Results are saved in:

`outputs/decision_tree_experiments.csv`

The experiments help compare how model settings affect prediction performance.

## Streamlit Application

The project includes an interactive web application where users can enter their educational information, skills, interests, and work preferences to receive a predicted career recommendation.

Run the application using:

`python -m streamlit run app.py`

## Project Structure

```text
Career_Recommendation_System/
│
├── dataset/
│   ├── Career_Path_Dataset_with_Career.csv
│   └── cleaned_career_dataset.csv
│
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── decision_tree_model.pkl
│
├── outputs/
│   ├── career_distribution.png
│   ├── missing_values.png
│   ├── age_distribution.png
│   ├── grade_distribution.png
│   ├── random_forest_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── model_comparison.csv
│   ├── random_forest_experiments.csv
│   └── decision_tree_experiments.csv
│
├── tests/
│   └── test_preprocessing.py
│
├── app.py
├── train_model.py
├── eda.py
├── check_categories.py
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/minahil-adnan/Career-Recommendation-System.git
```

### 2. Open the Project Folder

```bash
cd Career-Recommendation-System
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the Models

```bash
python train_model.py
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```

### 6. Run Tests

```bash
python -m pytest tests/test_preprocessing.py -v
```

## Testing

This project uses pytest for unit testing.

Four test cases are included:

1. Check that missing values are handled.
2. Check that the target column has no missing values.
3. Check that the dataset contains 22 career categories.
4. Check that the dataset contains no duplicate records.

All four tests passed during testing.

## Limitations

- Recommendations depend on the dataset used for training.
- Model predictions are not guaranteed career decisions.
- The dataset may not represent all students or real-world job market conditions.
- The model performance is modest, so predictions should be treated as suggestions rather than final decisions.
- More representative data and further experiments may improve performance.

## Conclusion

This project demonstrates how machine learning can be used for career-path prediction. It combines data preprocessing, exploratory data analysis, model training, hyperparameter experiments, model evaluation, and an interactive Streamlit application.