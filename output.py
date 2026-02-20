import streamlit as st
import pandas as pd
import pickle

# Load model & scaler
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("CreditCheck AI")
st.write("Credit Card Approval Prediction System")

# Inputs
gender = st.selectbox("Gender", ["M", "F"])
own_car = st.selectbox("Owns a Car", ["Y", "N"])
own_house = st.selectbox("Owns a House", ["Y", "N"])
income_type = st.selectbox(
    "Income Type",
    ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
)
education = st.selectbox(
    "Education Level",
    ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
)
family_status = st.selectbox(
    "Family Status",
    ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"]
)
housing_type = st.selectbox(
    "Housing Type",
    ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment"]
)

income = st.number_input("Annual Income", min_value=0)
children = st.number_input("Number of Children", min_value=0)
family_members = st.number_input("Family Members", min_value=1)
age = st.number_input("Age (Years)", min_value=18, max_value=100)
employed_years = st.number_input("Employment Duration (Years)", min_value=0)

# Manual encoding (same logic as LabelEncoder order)
input_df = pd.DataFrame([{
    "CODE_GENDER": 1 if gender == "M" else 0,
    "FLAG_OWN_CAR": 1 if own_car == "Y" else 0,
    "FLAG_OWN_REALTY": 1 if own_house == "Y" else 0,
    "CNT_CHILDREN": children,
    "AMT_INCOME_TOTAL": income,
    "NAME_INCOME_TYPE": ["Working","Commercial associate","Pensioner","State servant","Student"].index(income_type),
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special","Higher education","Incomplete higher","Lower secondary"].index(education),
    "NAME_FAMILY_STATUS": ["Single / not married","Married","Civil marriage","Separated","Widow"].index(family_status),
    "NAME_HOUSING_TYPE": ["House / apartment","Rented apartment","With parents","Municipal apartment","Office apartment"].index(housing_type),
    "FLAG_MOBIL": 1,
    "FLAG_WORK_PHONE": 0,
    "FLAG_PHONE": 0,
    "FLAG_EMAIL": 0,
    "CNT_FAM_MEMBERS": family_members,
    "AGE_YEARS": age,
    "EMPLOYED_YEARS": employed_years
}])

# SCALE FULL INPUT (now safe)
input_scaled = scaler.transform(input_df)

if st.button("Predict Credit Approval"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Risk Probability: {prob:.2f}")

    if prob < 0.30:
        st.success("✅ Approved (Low Risk)")
    else:
        st.error("❌ Rejected (High Risk)")
