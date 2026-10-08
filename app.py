import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Movie Rating - One-vs-Rest",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Rating Classification using One-vs-Rest")
st.write(
    "Predict whether a movie is Poor, Average, Good, or Excellent "
    "using One-vs-Rest Logistic Regression."
)

DATA_PATH = "movie_rating_ovr_dataset.csv"

FEATURES = [
    "IMDb_Rating",
    "Audience_Score",
    "Story_Score",
    "Production_Score"
]

TARGET = "Movie_Rating_Category"
CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
     
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model = OneVsRestClassifier(
        estimator=base_model
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    return (
        model,
        scaler,
        X_test,
        y_test,
        y_pred,
        y_proba
    )

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Keep movie_rating_ovr_dataset.csv "
        "in the same folder as this Streamlit file."
    )
    st.stop()

st.sidebar.header("Dataset Overview")
st.sidebar.metric("Number of Movies", len(df))
st.sidebar.metric("Number of Features", len(FEATURES))
st.sidebar.metric("Number of Classes", df[TARGET].nunique())

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

model, scaler, X_test, y_test, y_pred, y_proba = train_model(df)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("OvR Classifiers", len(model.estimators_))
col3.metric("Test Records", len(y_test))

st.subheader("Classification Report")
report = classification_report(
    y_test,
    y_pred,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(3), use_container_width=True)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
fig, ax = plt.subplots()
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_ORDER
)
display.plot(ax=ax, xticks_rotation=45)
st.pyplot(fig)

st.subheader("Predict a New Movie")
st.write("Adjust the movie features below and click Predict.")

col1, col2 = st.columns(2)

with col1:
    imdb_rating = st.slider(
        "IMDb Rating",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
        step=0.1
    )
    audience_score = st.slider(
        "Audience Score",
        min_value=0,
        max_value=100,
        value=90,
        step=1
    )

with col2:
    story_score = st.slider(
        "Story Score",
        min_value=0,
        max_value=100,
        value=85,
        step=1
    )
    production_score = st.slider(
        "Production Score",
        min_value=0,
        max_value=100,
        value=88,
        step=1
    )

new_movie = pd.DataFrame({
    "IMDb_Rating": [imdb_rating],
    "Audience_Score": [audience_score],
    "Story_Score": [story_score],
    "Production_Score": [production_score]
})

if st.button("Predict Movie Category", type="primary"):
    new_movie_scaled = scaler.transform(new_movie)

    prediction = model.predict(new_movie_scaled)[0]
    probabilities = model.predict_proba(new_movie_scaled)[0]

    st.success(f"Predicted Movie Category: {prediction}")

    probability_df = pd.DataFrame({
        "Category": model.classes_,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    st.subheader("Prediction Probabilities")
    st.dataframe(
        probability_df.style.format(
            {"Probability": "{:.2%}"}
        ),
        use_container_width=True
    )

    fig, ax = plt.subplots()
    ax.bar(
        probability_df["Category"],
        probability_df["Probability"]
    )
    ax.set_xlabel("Movie Rating Category")
    ax.set_ylabel("Probability")
    ax.set_title("OvR Prediction Probabilities")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

