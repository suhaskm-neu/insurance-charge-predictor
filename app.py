import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model components
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

# Page configuration
st.set_page_config(
    page_title="Insurance Charge Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 🧾 App Title and Introduction
st.title("💸 Insurance Charge Predictor")
st.markdown("""
    This application predicts insurance charges based on personal and demographic factors.
    Fill in your information below to get an estimate of your insurance charges.
    
    ---
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Personal Information")
    age = st.slider(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        help="Select your age (18-80 years)"
    )
    
    sex = st.selectbox(
        "Sex",
        ["male", "female"],
        help="Select your biological sex"
    )
    
    bmi = st.slider(
        "BMI (Body Mass Index)",
        min_value=15.0,
        max_value=50.0,
        value=25.0,
        help="BMI is a measure of body fat based on height and weight. Normal range is 18.5-24.9"
    )

with col2:
    st.subheader("🏠 Additional Details")
    children = st.slider(
        "Number of Children/Dependents",
        min_value=0,
        max_value=5,
        value=0,
        help="Number of children covered by insurance"
    )
    
    smoker = st.selectbox(
        "Smoking Status",
        ["no", "yes"],
        help="Current smoking status - significant factor in insurance pricing"
    )
    
    region = st.selectbox(
        "Geographic Region",
        ["southeast", "southwest", "northeast", "northwest"],
        help="Your residence region in the United States"
    )

st.markdown("---")

# 🔄 Preprocess input
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
}

# Ensure all expected features are present
for feature in model_features:
    if feature not in input_dict:
        input_dict[feature] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])[model_features]

# Scale numeric columns
input_df[["age", "bmi", "children"]] = scaler.transform(input_df[["age", "bmi", "children"]])

# 🎯 Make Prediction
if st.button("Calculate Insurance Charges 💰", help="Click to predict your insurance charges"):
    with st.spinner("Calculating..."):
        prediction = model.predict(input_df)[0]
        
    st.markdown("### 📝 Prediction Results")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.success(f"### Estimated Annual Insurance Charges:\n## ${prediction:,.2f}")
        
    st.info("""
        **Note:** This prediction is based on historical data and should be used as a general estimate only.
        Actual insurance charges may vary based on additional factors and specific insurance provider policies.
    """)

# Add footer with additional information
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
    <small>This tool uses machine learning to predict insurance charges based on demographic and health factors.
    Results are estimates and should not be considered as final quotes.</small>
    </div>
""", unsafe_allow_html=True)

### Simpler UI Template

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load model components
# model = joblib.load("best_model.pkl")
# scaler = joblib.load("scaler.pkl")
# model_features = joblib.load("model_features.pkl")

# st.set_page_config(page_title="Insurance Charge Predictor", layout="centered")

# # 🧾 App Title
# st.title("💸 Insurance Charge Predictor")

# # 🧍 User Inputs
# age = st.slider("Age", 18, 80, 30)
# sex = st.selectbox("Sex", ["male", "female"])
# bmi = st.slider("BMI", 15.0, 50.0, 25.0)
# children = st.slider("Number of Children", 0, 5, 0)
# smoker = st.selectbox("Smoker?", ["yes", "no"])
# region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# # 🔄 Preprocess input
# input_dict = {
#     "age": age,
#     "bmi": bmi,
#     "children": children,
#     "sex_male": 1 if sex == "male" else 0,
#     "smoker_yes": 1 if smoker == "yes" else 0,
#     "region_northwest": 1 if region == "northwest" else 0,
#     "region_southeast": 1 if region == "southeast" else 0,
#     "region_southwest": 1 if region == "southwest" else 0
# }

# # Ensure all expected features are present
# for feature in model_features:
#     if feature not in input_dict:
#         input_dict[feature] = 0

# # Convert to DataFrame
# input_df = pd.DataFrame([input_dict])[model_features]

# # Scale numeric columns
# input_df[["age", "bmi", "children"]] = scaler.transform(input_df[["age", "bmi", "children"]])

# # 🎯 Make Prediction
# if st.button("Predict Charges 💰"):
#     prediction = model.predict(input_df)[0]
#     st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")
