import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load everything
reg_model = joblib.load('student_reg_model.pkl')
clf_model = joblib.load('student_clf_model.pkl')
encoders = joblib.load('encoders.pkl')
target_le = joblib.load('target_le.pkl')
edu_map = joblib.load('edu_map.pkl')
features_list = joblib.load('features_list.pkl')

st.title("🎓 Student Performance Predictor")

# Sidebar inputs
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
race = st.sidebar.selectbox("Race", encoders['race/ethnicity'].classes_)
parent_edu = st.sidebar.selectbox("Parent Education", list(edu_map.keys()))
lunch = st.sidebar.selectbox("Lunch", encoders['lunch'].classes_)
prep = st.sidebar.selectbox("Test Prep", encoders['test preparation course'].classes_)
hours = st.sidebar.slider("Study Hours", 5, 40, 15)

if st.button("Predict"):
    # Create input dataframe
    input_df = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': race,
        'lunch': lunch,
        'test preparation course': prep,
        'parental level of education': parent_edu,
        'study_hours': hours,
        'edu_score': edu_map[parent_edu],
        'prep_done': 1 if prep == 'completed' else 0
    }])
    input_df['study_index'] = input_df['edu_score'] + input_df['prep_done']
    
    # Encode inputs using saved encoders
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])
    
    # Match feature order
    input_df = input_df[features_list]
    
    # Predict
    scores = reg_model.predict(input_df)[0]
    cat_idx = clf_model.predict(input_df)[0]
    category = target_le.inverse_transform([cat_idx])[0]
    
    # Display
    c1, c2, c3 = st.columns(3)
    c1.metric("Math", int(scores[0]))
    c2.metric("Reading", int(scores[1]))
    c3.metric("Writing", int(scores[2]))
    
    st.subheader(f"Category: {category}")
    if category == "At Risk":
        st.error("Recommendation: Immediate Tutoring Required.")
    else:
        st.success("Recommendation: Keep up the good work!")
