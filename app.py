import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Pima Diabetes Prediction")

# Input fields for features
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("BloodPressure")
skin_thickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", min_value=0)

# Define column names exactly as in training
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
           'Insulin','BMI','DiabetesPedigreeFunction','Age']

# When button is clicked
if st.button("Predict"):
    # Convert input to DataFrame with same column names
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]],
                        columns=columns)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Display results
    if prediction[0] == 1:
        st.error(f"Predicted: Diabetes positive (Risk: {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"Predicted: Diabetes negative (Risk: {probability[0][1]*100:.2f}%)")
