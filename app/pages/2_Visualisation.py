import os
import pickle as pkl
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get current file directory
current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Construct absolute paths
pickle_file_path = os.path.join(root_dir, "pickle", "ModleComparisonResults.pkl")
csv_file_path = os.path.join(root_dir, "Salary_Data.csv")

# Load data
with open(pickle_file_path, "rb") as file:
    comparison_df = pkl.load(file)

data = pd.read_csv(csv_file_path)


st.markdown("# Why :rainbow[XGBoost ]?")
st.markdown("---")



with open("./../pickle/ModleComparisonResults.pkl", "rb") as file:
  model = pkl.load(file)



card_style = """
<style>
/* --- Grid Layout for Cards --- */
.cards {
    display: grid;
    grid-template-columns: repeat(4, minmax(250px, 1fr));
    gap: 3%;
    margin-top: 20px;
}

/* --- Individual Card Styling --- */
.card {
    background-color:#1b1c1c;// #222; /* Dark background for contrast */
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: #fff;
    height:80%;
    border: 1px solid #444;
    
    /* --- Key properties for the effect --- */
    position: relative; /* Establishes a positioning context */
    overflow: hidden;   /* Hides the pseudo-element when it's outside */
    z-index: 1;         /* Ensures card is a stacking context */
}

/* --- The Water Flow Pseudo-Element --- */
.card::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 0; /* Starts with zero height */
    background: linear-gradient(to top, #3498db, #8e44ad); /* The gradient */
    z-index: -1; /* Positioned behind the content */
    
    /* --- The Animation --- */
    transition: height 0.4s ease-in-out;
}

/* --- Triggering the animation on hover --- */
.card:hover::before {
    height: 100%; /* Fills the card on hover */
}

/* --- Ensuring Content Stays on Top --- */
.card p, .card h3 {
    position: relative;
    z-index: 2; /* Keeps text above the ::before element */
    margin: 5px 0;
}
.card h3{
color:#43fa00;}

.card span{
font-size:1.5rem;}




</style>

"""

card_html =""
for index,row in model.iterrows():
  accuracy = row["Accuracy"]
  card_html += f"""
  <div class="card">
  <p><span>{row["Model"]}</span></p>
  <p>Accuracy: {accuracy:.4f}</p>
   <h3>{accuracy*100:.1f}%<h3>
  
  </div>
  """
card_html = "<div class='cards'>"+card_html+"</div>"
st.markdown(card_style+card_html,unsafe_allow_html=True)



st.markdown("<h2> XGBoost Stands out with <span style='color:green'>95%</span> of accuracy</h2>", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Top 5 High Paid Jobs")
highpaid = data.sort_values(by="Salary", ascending=False).head(5).reset_index()
st.write(highpaid)

st.subheader("Top 5 Low Paid Jobs")
st.write(highpaid.tail(5).reset_index())

st.markdown("---")

st.subheader("Experience vs Salary")
col1, col2 = st.columns(2)
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(data=data, x="Years of Experience", y="Salary", hue="Gender", ax=ax)
legend = ax.legend()
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
for text in legend.get_texts():
    text.set_color('black')
st.pyplot(fig)

st.markdown("---")

st.subheader("Correlation of variables")
corr = data.select_dtypes(include=["number"]).corr()
fig1, ax1 = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, ax=ax1, cmap="Blues")
st.pyplot(fig1)
st.markdown("---")


st.subheader("Gender vs Salary")
fig2, ax2 = plt.subplots(figsize=(15, 10))
sns.barplot(data, x="Gender",y="Salary", ax=ax2,color="lightblue")
st.pyplot(fig2)


