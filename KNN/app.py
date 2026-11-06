import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("knn_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Cancer Prediction System")
st.write("Machine Learning Model: K-Nearest Neighbors")

st.subheader("Enter Tumor Measurements")

radius = st.number_input("Mean Radius", 0.0, 30.0, 14.0)
texture = st.number_input("Mean Texture", 0.0, 40.0, 20.0)
perimeter = st.number_input("Mean Perimeter", 0.0, 200.0, 90.0)
area = st.number_input("Mean Area", 0.0, 3000.0, 600.0)
smoothness = st.number_input("Mean Smoothness", 0.0, 1.0, 0.1)

# Create full feature array (30 features expected)
input_data = np.zeros((1, 30))
input_data[0][0] = radius
input_data[0][1] = texture
input_data[0][2] = perimeter
input_data[0][3] = area
input_data[0][4] = smoothness

# Scale input
scaled_input = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(scaled_input)

    if prediction[0] == 0:
        st.write("Prediction: Malignant Tumor (Cancer Detected)")
    else:
        st.write("Prediction: Benign Tumor (No Cancer Detected)")