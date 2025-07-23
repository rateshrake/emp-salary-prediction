import streamlit as st
import pandas as pd
import os

current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, "..", "..","Salary_Data.csv"))
st.header("About!")


st.markdown(
    """
    <div style="background-color:#f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #ccc;">
        <h2 style="color:#1e3d59; font-family: Arial, sans-serif;">📊 Dataset Source</h2>
        <p style="font-size:16px; color:#333;">The dataset is taken from GitHub and can be accessed here:</p>
        <a href="https://github.com/Pranjali1049/Salary_Prediction/blob/main/Salary_Data.csv" 
           target="_blank" 
           style="font-size:16px; color:#007acc; text-decoration:none; font-weight:bold;">
           🔗 Salary_Data.csv on GitHub
        </a>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    '---'
)

st.markdown(
    """
    <div style="background-color:#f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #ccc;">
        <h2 style="color:#1e3d59; font-family: Arial, sans-serif;">📌 Models Used</h2>
        <p style="font-size:16px; color:#333; font-family: Arial;">We tested a variety of models to compare their performance:</p>
        <ul style="font-size:16px; color:#333; font-family: Arial; line-height: 1.6;">
            <li>XGBoost</li>
            <li>Linear Regression</li>
            <li>K-Nearest Neighbors (KNN)</li>
            <li>Decision Tree</li>
            <li>Gradient Boosting</li>
            <li>Logistic Regression</li>
            <li>Naive Bayes</li>
            <li>Support Vector Machine (SVM)</li>
            <li>Random Forest</li>
        </ul>
        <h3 style="color:black;"> out of all the models we get high accuracy for the<tt> XGBoost</tt ></h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

dataset=  pd.read_csv(data_dir)
st.dataframe(dataset)

st.markdown("---")

st.markdown(
    """
    <div style="background-color:#f7f9fc; padding: 25px; border-radius: 10px; border: 1px solid #ccc;">
        <h1 style="color:#2c3e50; font-family: Georgia, serif;">💼 Employee Salary Prediction App</h1>
        <p style="color:black;">
            The data has the following columns The data has the following columns:</p>
            <li style="color:black;"><strong>Age, Gender, Education Level,Job Title, Years of Experience, Salary</strong></li>
            <li style="color:black;">✔️ We used <strong>Label Encoding</strong> to convert categorical data to numerical format.</li>
            <li style="color:black;">✔️ We applied <strong>StandardScaler</strong> for feature scaling before model training.</li>
        </p>
       
    </div>
    """,
    unsafe_allow_html=True
)



