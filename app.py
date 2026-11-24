
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# Load trained Logistic Regression model
@st.cache_resource
def load_model():
    return joblib.load("models/logistic_regression_model.pkl")


try:
    model = load_model()
except Exception as error:
    st.error(f"Model Loading Error: {error}")
    st.stop()


# Main heading
st.title("🎓 Career Recommendation System")

st.write(
    "Find a suitable career path based on your "
    "education, skills, interests and preferences."
)

st.divider()


# User input form
with st.form("career_form"):

    st.header("Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=60,
            value=20,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        location = st.text_input(
            "Location",
            "Peshawar"
        )

        qualification = st.selectbox(
            "Highest Qualification",
            ["B.Tech", "B.Sc", "12th", "MBA",
             "M.Sc", "MCA", "M.Tech", "10th"]
        )

        stream = st.selectbox(
            "Academic Stream",
            ["ECE", "BBA", "Civil", "Mechanical",
             "EEE", "IT", "Commerce", "CSE"]
        )

    with col2:

        academic_level = st.selectbox(
            "Current Academic Level",
            ["Student", "Working Professional", "Graduate"]
        )

        grade = st.number_input(
            "CGPA / Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.5
        )

        work_style = st.selectbox(
            "Preferred Work Style",
            ["Remote", "Office", "Field Work", "Hybrid"]
        )

        work_type = st.selectbox(
            "Work Type Interest",
            ["Technical", "Creative", "Social", "Analytical"]
        )

        willing_to_relocate = st.selectbox(
            "Willing to Relocate?",
            ["Yes", "No"]
        )

    st.divider()

    st.header("Skills and Interests")

    col3, col4 = st.columns(2)

    with col3:

        technical_skills = st.text_input(
            "Technical Skills",
            "Python, C++, HTML/CSS"
        )

        soft_skills = st.text_input(
            "Soft Skills",
            "Communication, Teamwork"
        )

        languages = st.text_input(
            "Languages Known",
            "English, Urdu"
        )

        certifications = st.text_input(
            "Certifications",
            "Not Provided"
        )

    with col4:

        interests = st.text_input(
            "Fields of Interest",
            "Artificial Intelligence, Programming"
        )

        past_jobs = st.text_input(
            "Past Jobs / Internships",
            "Not Provided"
        )

        achievements = st.text_input(
            "Achievements",
            "Not Provided"
        )

    submit = st.form_submit_button(
        "🔍 Predict My Career",
        use_container_width=True
    )


# Career prediction
if submit:

    # Input validation
    required_fields = {
        "Location": location,
        "Technical Skills": technical_skills,
        "Soft Skills": soft_skills,
        "Languages Known": languages,
        "Fields of Interest": interests
    }

    empty_fields = [
        field for field, value in required_fields.items()
        if not value.strip()
    ]

    if empty_fields:

        st.warning(
            "Please fill in these required fields: "
            + ", ".join(empty_fields)
        )

    else:

        # Prepare input data
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Location": location.strip(),
            "Highest_qualification": qualification,
            "Stream": stream,
            "Current_Academic_Level": academic_level,
            "Grade_CGPA_Percentage": grade,
            "Technical_Skills": technical_skills.strip(),
            "Soft_Skills": soft_skills.strip(),
            "Languages_Known": languages.strip(),
            "Certifications": certifications.strip() or "Not Provided",
            "Fields_of_Interest": interests.strip(),
            "Preferred_Work_Style": work_style,
            "Work_Type_Interest": work_type,
            "Past_Jobs_Internships": past_jobs.strip() or "Not Provided",
            "Achievements": achievements.strip() or "Not Provided",
            "Willing_to_Relocate": willing_to_relocate
        }])

        try:

            # Predict career
            prediction = model.predict(input_data)

            st.divider()
            st.success("Career Prediction Completed!")

            st.subheader("Your Recommended Career")

            st.title(f"🎯 {prediction[0]}")

            st.info(
                "This recommendation is generated by a "
                "machine learning model. It is a suggestion, "
                "not a guaranteed career decision."
            )

        except Exception as error:

            st.error(f"Prediction Error: {error}")
