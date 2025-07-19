import streamlit as st
import pandas as pd
import joblib
import random

model = joblib.load("xg_pipeline.pkl")
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# Remove default padding from the top for title
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    """
    <h1 style='
        text-align: center;
        color: #0d6efd;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        margin-top: 0px;
        margin-bottom: 60px;
    '>
        💼 Employee Salary Predictor
    </h1>
    """,
    unsafe_allow_html=True,
)

# Columns
left_col, right_col = st.columns([1, 1], gap="large")

#Basic Info
with left_col:
    st.markdown(
        """
        <div style="
            padding: 25px;
            margin-bottom: 40px;
            border-radius: 15px;
            background-color: #f0f4f8;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        ">
        <h2 style="
            color: #0d6efd;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 30px;
        ">
            📊 Basic Info
        </h2>
        """,
        unsafe_allow_html=True,
    )

    age = st.slider("🧓 Age", 18, 70, 30)
    hours_per_week = st.slider("⏱️ Hours per Week", 5, 70, 40)

    predict_now = st.button("🔍 Predict Income")
    st.markdown("</div>", unsafe_allow_html=True)

#Other Details
with right_col:
    st.markdown(
        """
        <div style="
            padding: 25px;
            margin-bottom: 40px;
            border-radius: 15px;
            background-color: #f0f4f8;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        ">
        <h2 style="
            color: #0d6efd;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 30px;
        ">
            📋 Other Details
        </h2>
        """,
        unsafe_allow_html=True,
    )

    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    education_levels = {
        "HS-grad": 9,
        "Some-college": 10,
        "Assoc-acdm": 11,
        "Assoc-voc": 12,
        "Bachelors": 13,
        "Masters": 14,
        "Prof-school": 15,
        "Doctorate": 16
    }
    education_cat = st.selectbox("🎓 Education Level", list(education_levels.keys()))
    education_num = education_levels[education_cat]

    occupation = st.selectbox(
        "🛠️ Occupation",
        [
            "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
            "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
            "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
            "Armed-Forces", "Others"
        ],
    )

    native_country = st.selectbox(
        "🌍 Native Country",
        ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Other"]
    )
    st.markdown("</div>", unsafe_allow_html=True)


defaults = {
    "fnlwgt": 123456,
    "workclass": "Private",
    "marital-status": "Never-married",
    "capital-gain": 0,
    "capital-loss": 0,
    "race": "White",
}

# Prediction
if predict_now:
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "occupation": [occupation],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country],
        "fnlwgt": [defaults["fnlwgt"]],
        "educational-num": [education_num],
        "workclass": [defaults["workclass"]],
        "marital-status": [defaults["marital-status"]],
        "capital-gain": [defaults["capital-gain"]],
        "capital-loss": [defaults["capital-loss"]],
        "race": [defaults["race"]],
    })

    prediction = model.predict(input_df)[0]

    if prediction == 0:
        income_class = "<=50K"
        est_income_val = random.randint(20000, 50000)
    else:
        income_class = ">50K"
        est_income_val = random.randint(55000, 130000)

    est_income = f"${est_income_val:,}"

    st.success(f"✅ Predicted Income Class: **{income_class}**")
    st.info(f"💵 Estimated Salary: **{est_income}**")