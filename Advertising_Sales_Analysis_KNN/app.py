import streamlit as st
import pandas as pd
import numpy as np
import pickle

encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler_knn.pkl", "rb"))
model = pickle.load(open("knn_sales_model.pkl", "rb"))

df = pd.read_csv("Advertising.csv")

CATEGORICAL_COLUMNS = [
    "Region", "Market_Type", "Season",
    "Customer_Segment", "Campaign_Type",
    "Ad_Keyword", "Feedback_Tag"
]

NUMERICAL_COLUMNS = [
    "TV", "Radio", "Newspaper",
    "Discount_Percent", "Ad_Frequency"
]

st.title("Advertising Sales Prediction (KNN)")

input_data = {}

for col in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS:
    if col in CATEGORICAL_COLUMNS:
        input_data[col] = st.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.number_input(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

input_df = pd.DataFrame([input_data])

if st.button("Predict Sales"):

    X_cat = encoder.transform(
        input_df[CATEGORICAL_COLUMNS]
    ).toarray()

    X_num = scaler.transform(
        input_df[NUMERICAL_COLUMNS].values
    )

    X_final = np.hstack((X_cat, X_num))

    prediction = model.predict(X_final)

    st.success(f"Predicted Sales: {int(prediction[0])}")
