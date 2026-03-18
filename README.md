Student Performance AI System
-----------------------------
Project Overview:
An AI system designed to predict student scores (Math, Reading, Writing) and categorize 
performance into 'At Risk', 'Average', or 'High Performer'.

Model Performance:
- Regression (Random Forest): R2 > 0.85 (with study hours feature)
- Classification (Random Forest): Accuracy > 90%

Key Features:
1. Multi-Output Regression for simultaneous score prediction.
2. Feature engineering: Study Index, Education Scoring.
3. Interactive Streamlit dashboard for students and teachers.
4. Bias Detection: Analyzes the impact of lunch type (socio-economic) and parental education.

How to Run:
1. Run 'python train_model.py' to generate models.
2. Run 'streamlit run student_app.py' to launch the UI.
