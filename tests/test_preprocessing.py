import pandas as pd


def test_missing_values_are_handled():
    data = {
        "Certifications": [None, "Python"],
        "Achievements": [None, "Award"],
        "Past_Jobs_Internships": [None, "Internship"]
    }

    df = pd.DataFrame(data)

    df["Certifications"] = df["Certifications"].fillna("Not Provided")
    df["Achievements"] = df["Achievements"].fillna("Not Provided")
    df["Past_Jobs_Internships"] = df["Past_Jobs_Internships"].fillna("Not Provided")

    assert df.isnull().sum().sum() == 0


def test_dataset_has_no_empty_target():
    df = pd.read_csv("dataset/cleaned_career_dataset.csv")

    assert df["Suggested_Career_Path"].isnull().sum() == 0


def test_dataset_has_22_career_categories():
    df = pd.read_csv("dataset/cleaned_career_dataset.csv")

    assert df["Suggested_Career_Path"].nunique() == 22


def test_dataset_has_no_duplicate_records():
    df = pd.read_csv("dataset/cleaned_career_dataset.csv")

    assert df.duplicated().sum() == 0