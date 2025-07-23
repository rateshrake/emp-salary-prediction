import pickle as pkl
import streamlit as st
import pandas as pd
import time
import random


st.set_page_config(page_title="XGBoost Salary App", layout="wide")
with open("./../pickle/XGBoost_Model.pkl", "rb") as file:
    model = pkl.load(file)

with open("./../pickle/Encoder.pkl","rb") as file:
    encoder = pkl.load(file)


dataset = pd.read_csv("./../Salary_Data.csv")
dataset.dropna(inplace=True)

st.session_state.dataset = dataset



st.sidebar.markdown(
    "<h1 style='color:white; font-weight:bold;font-size:2.5rem'>Experiment🧪</h1>",
    unsafe_allow_html=True
)
st.sidebar.header("Enter Candidate Information:")

age = st.sidebar.slider("Age", 20, 60, 36)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
education = st.sidebar.selectbox("Education Level", list(dataset["Education Level"].unique()))
job = st.sidebar.selectbox("Job Title", list(dataset["Job Title"].unique()))
experience = st.sidebar.slider("Experience (in years)", 0, int(dataset["Years of Experience"].max()), 3)
st.subheader("for bulk predictions:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job],
    "Years of Experience": [experience]
})


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully.")
    df.dropna(inplace=True)

else:

    df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job],
        "Years of Experience": [experience]
    })


st.subheader("🔢Input Data")
st.dataframe(df)


encoded_df = df.copy()
for col, le in encoder.items():
    if col in encoded_df.columns:
        encoded_df[col] = le.transform(encoded_df[col])



predicted_salary = model.predict(encoded_df)
pr =  encoder["Salary"].inverse_transform(predicted_salary)



st.markdown('---')
st.subheader("Predicted Salary:")
if uploaded_file:
    st.write(pr)
else:
    st.markdown(
        f"<h2 style='color:green; font-weight:bold;'>💰₹ {pr[0]}</h2>",
        unsafe_allow_html=True
    )

st.markdown('---')
st.table(encoder["Salary"].classes_) 
