import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(" ", "_")
    df.drop_duplicates(inplace=True)
    return df

def feature_engineering(df):
    df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
    df["average_score"] = df["total_score"] / 3

    edu_map = {
        "some high school": 0,
        "high school": 1,
        "some college": 2,
        "associate's degree": 3,
        "bachelor's degree": 4,
        "master's degree": 5
    }

    df["study_index"] = df["test_preparation_course"].map({
        "none": 0,
        "completed": 1
    }) + df["parental_level_of_education"].map(edu_map)

    df["performance_category"] = pd.cut(
        df["average_score"],
        bins=[0, 50, 75, 100],
        labels=["Low", "Medium", "High"]
    )

    return df

def encode(df):
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def train_model(df):
    X = df.drop(["math_score", "reading_score", "writing_score", "performance_category"], axis=1)
    y = df[["math_score", "reading_score", "writing_score"]]

    X, encoders = encode(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear": MultiOutputRegressor(LinearRegression()),
        "RandomForest": MultiOutputRegressor(RandomForestRegressor()),
        "GradientBoost": MultiOutputRegressor(GradientBoostingRegressor()),
        "XGBoost": MultiOutputRegressor(XGBRegressor())
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = model

    with open("model.pkl", "wb") as f:
        pickle.dump((best_model, encoders), f)

    return best_model, best_score

def classify_risk(avg):
    if avg < 50:
        return "At Risk"
    elif avg < 75:
        return "Average"
    else:
        return "High Performer"

def predict_student(input_data):
    model, encoders = pickle.load(open("model.pkl", "rb"))

    df = pd.DataFrame([input_data])

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    preds = model.predict(df)[0]
    avg = np.mean(preds)

    category = classify_risk(avg)

    suggestions = []
    if preds[0] < preds[1]:
        suggestions.append("Focus more on Math practice")
    if input_data["test_preparation_course"] == "none":
        suggestions.append("Complete test preparation course")
    if avg < 60:
        suggestions.append("Increase study hours")

    return {
        "math": preds[0],
        "reading": preds[1],
        "writing": preds[2],
        "average": avg,
        "category": category,
        "suggestions": suggestions
    }

def generate_insights(df):
    insights = {}

    insights["test_prep"] = df.groupby("test_preparation_course")[["math_score","reading_score","writing_score"]].mean()
    insights["gender"] = df.groupby("gender")[["math_score","reading_score","writing_score"]].mean()
    insights["lunch"] = df.groupby("lunch")[["math_score","reading_score","writing_score"]].mean()
    insights["parent_edu"] = df.groupby("parental_level_of_education")[["math_score","reading_score","writing_score"]].mean()

    return insights

if __name__ == "__main__":
    df = load_data("StudentsPerformance.csv")
    df = feature_engineering(df)

    model, score = train_model(df)
    print("Best R2 Score:", score)

    sample = {
        "gender": "female",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "study_hours": 3
    }

    result = predict_student(sample)

    print("Prediction:", result)

    insights = generate_insights(df)
    print("Insights Generated")
