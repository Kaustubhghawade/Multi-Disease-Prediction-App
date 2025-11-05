import streamlit as st
import pickle
import pandas as pd

# ------------------------
# Helper function to load model
# ------------------------
def load_model(model_name):
    with open(f"models/{model_name}_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["features"]

# ------------------------
# Predict function
# ------------------------
def predict_disease(model_name, input_data):
    model, scaler, features = load_model(model_name)
    
    # Ensure all features exist in input_data
    for f in features:
        if f not in input_data:
            input_data[f] = 0  # default value if missing

    df = pd.DataFrame([input_data], columns=features)
    
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data).max() * 100
    return prediction, prob


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Multi-Disease Prediction", layout="centered")
st.title("🩺 Multi-Disease Prediction App")

# Disease selection
disease = st.selectbox(
    "Select Disease",
    ("Diabetes", "Heart Disease", "Parkinson's Disease")
)

st.subheader(f"Enter your health data for {disease}:")

# ------------------------
# Input fields for each disease
# ------------------------
user_input = {}

if disease == "Diabetes":
    user_input["Pregnancies"] = st.number_input("Pregnancies", 0, 20, 1)
    user_input["Glucose"] = st.number_input("Glucose", 0, 200, 120)
    user_input["BloodPressure"] = st.number_input("BloodPressure", 0, 140, 70)
    user_input["SkinThickness"] = st.number_input("SkinThickness", 0, 100, 20)
    user_input["Insulin"] = st.number_input("Insulin", 0, 900, 80)
    user_input["BMI"] = st.number_input("BMI", 0.0, 70.0, 25.0)
    user_input["DiabetesPedigreeFunction"] = st.number_input("DiabetesPedigreeFunction", 0.0, 2.5, 0.5)
    user_input["Age"] = st.number_input("Age", 0, 120, 30)

elif disease == "Heart Disease":
    user_input["age"] = st.number_input("Age", 0, 120, 45)
    user_input["sex"] = st.number_input("Sex (1=Male, 0=Female)", 0, 1, 1)
    user_input["cp"] = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
    user_input["trestbps"] = st.number_input("Resting Blood Pressure", 80, 200, 120)
    user_input["chol"] = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    user_input["fbs"] = st.number_input("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", 0, 1, 0)
    user_input["restecg"] = st.number_input("Resting ECG (0-2)", 0, 2, 1)
    user_input["thalach"] = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    user_input["exang"] = st.number_input("Exercise Induced Angina (1=Yes,0=No)", 0, 1, 0)
    user_input["oldpeak"] = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0)
    user_input["slope"] = st.number_input("Slope of peak exercise ST segment (0-2)", 0, 2, 1)
    user_input["ca"] = st.number_input("Number of major vessels colored by flourosopy (0-3)", 0, 3, 0)
    user_input["thal"] = st.number_input("Thal (1 = normal; 2 = fixed defect; 3 = reversable defect)", 1, 3, 2)

elif disease == "Parkinson's Disease":
    # Features from dataset (excluding 'name' and 'status')
    parkinsons_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "MDVP:DDP", "MDVP:Shimmer",
        "MDVP:Shimmer(dB)", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2",
        "D2", "PPE"
    ]
    for feature in parkinsons_features:
        user_input[feature] = st.number_input(feature, 0.0, 500.0, 0.5)

# ------------------------
# Prediction button
# ------------------------
if st.button("Predict"):
    # Map disease selection to actual model filenames
    disease_to_model = {
        "Diabetes": "diabetes",
        "Heart Disease": "heart",
        "Parkinson's Disease": "parkinsons"
    }
    model_name = disease_to_model[disease]

    prediction, prob = predict_disease(model_name, user_input)

    # Show results
    st.subheader("Prediction:")
    if disease == "Diabetes":
        st.write("Patient has Diabetes" if prediction==1 else "Patient is Healthy")
    elif disease == "Heart Disease":
        st.write("Patient at Risk of Heart Disease" if prediction==1 else "Patient is Healthy")
    else:
        st.write("Patient has Parkinson's Disease" if prediction==1 else "Patient is Healthy")

    st.write(f"Prediction Confidence: {prob:.2f}%")
