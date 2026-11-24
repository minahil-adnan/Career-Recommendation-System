
# Career Recommendation System

## Project Overview
The Career Recommendation System is a Machine Learning project that predicts a suitable career path based on a person's educational background, technical skills, interests, and work preferences.

The system uses supervised machine learning classification models to recommend a career category.

## Objectives
- Analyze career-related data.
- Clean and preprocess the dataset.
- Explore data through visualizations.
- Train and compare machine learning models.
- Build an interactive Streamlit application.

## Dataset
**Dataset Name:** Career Path Dataset with Career

**Source:** [Mendeley Data](https://data.mendeley.com/datasets/cns7h4b9g7/1)

- Total Records: 15,000
- Total Columns: 19
- Target Column: `Suggested_Career_Path`
- Career Categories: 22

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Data Preprocessing
- Checked missing values.
- Handled missing values in selected columns.
- Checked duplicate records.
- Separated features and target.
- Applied One-Hot Encoding to categorical features.
- Split data into training and testing sets.

## Exploratory Data Analysis (EDA)
The following visualizations were created:

1. Career Distribution
2. Missing Values
3. Age Distribution
4. Grade Distribution

## Machine Learning Models
Two supervised classification models were implemented:

1. Random Forest Classifier
2. Logistic Regression

Both models were evaluated and compared using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Results
The models are saved in the `models` folder.

The model comparison results are available in:

`outputs/model_comparison.csv`

## Streamlit Application
The project includes an interactive web application where users can enter their educational information, skills, interests, and work preferences to receive a predicted career recommendation.

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
│   └── logistic_regression_model.pkl
│
├── outputs/
│   ├── career_distribution.png
│   ├── missing_values.png
│   ├── age_distribution.png
│   ├── grade_distribution.png
│   ├── random_forest_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   └── model_comparison.csv
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
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd Career_Recommendation_System
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
streamlit run app.py
```

## Limitations
- Recommendations depend on the dataset used for training.
- Model predictions are not guaranteed career decisions.
- The dataset may not represent all students or real-world job market conditions.
- Model performance can be improved with more representative data and further experiments.

## Testing

This project uses pytest for unit testing.

Three test cases are included:

1. Check that missing values are handled.
2. Check that the target column has no missing values.
3. Check that the dataset contains 22 career categories.

Run tests using:

```bash
python -m pytest tests/test_preprocessing.py -v

## Conclusion
This project demonstrates how machine learning can be used for career-path prediction. It combines data preprocessing, exploratory data analysis, model training, evaluation, and an interactive Streamlit application.