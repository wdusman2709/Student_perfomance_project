import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and artifacts
reg_model = joblib.load('student_reg_model.pkl')
clf_model = joblib.load('student_clf_model.pkl')
model_cols = joblib.load('model_columns.pkl')
perf_classes = joblib.load('perf_classes.pkl')
edu_map = joblib.load('edu_map.pkl')

st.set_page_config(page_title="Student AI Success System", layout="wide")
st.title("🎓 Student Exam Performance Prediction & Insights")

# Sidebar - Inputs
st.sidebar.header("Student Demographics & Habits")
gender = st.sidebar.selectbox("Gender", ["female", "male"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.sidebar.selectbox("Parental Education", list(edu_map.keys()))
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
study_hours = st.sidebar.slider("Weekly Study Hours", 0, 40, 15)

# Preprocessing Input
input_data = pd.DataFrame([{
    'gender': gender, 'race/ethnicity': race, 'lunch': lunch, 
    'test preparation course': test_prep, 'parental level of education': parent_edu
}])
input_encoded = pd.get_dummies(input_data)
# Add missing columns with 0
for col in model_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Calculated numeric features
input_encoded['study_hours'] = study_hours
input_encoded['parental_edu_score'] = edu_map[parent_edu]
input_encoded['prep_score'] = 1 if test_prep == 'completed' else 0
input_encoded['study_index'] = input_encoded['parental_edu_score'] + input_encoded['prep_score']
input_encoded = input_encoded[model_cols] # Reorder

# Predictions
if st.button("Predict Performance"):
    scores = reg_model.predict(input_encoded)[0]
    category_idx = clf_model.predict(input_encoded)[0]
    category = perf_classes[category_idx]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Math Score", f"{int(scores[0])}")
    col2.metric("Reading Score", f"{int(scores[1])}")
    col3.metric("Writing Score", f"{int(scores[2])}")
    
    avg_score = np.mean(scores)
    st.subheader(f"Performance Category: {category}")
    
    # Recommendations
    st.write("### 💡 Actionable Insights")
    if category == "At Risk":
        st.error("Priority Intervention: Schedule mandatory tutoring sessions.")
    elif category == "Average":
        st.warning("Strategy: Focus on Math practice (+10 score improvement possible).")
    else:
        st.success("High Performer: Suggested for Advanced Placement (AP) courses.")
        
    if test_prep == 'none':
        st.info("💡 Completing a test prep course could increase scores by ~15%.")

# Analytics Section
st.divider()
st.subheader("📊 Institutional Performance Analytics")
st.write("Analyzing impact of socio-economic factors on success.")
# (Insert logic here to display bias detection or demographic charts)

