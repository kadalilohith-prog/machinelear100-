import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "movie_rating_ovr_dataset.csv"

FEATURES = [
    "IMDb_Rating",
    "Audience_Score",
    "Story_Score",
    "Production_Score"
]

TARGET = "Movie_Rating_Category"
CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# 2. Select features and target
X = df[FEATURES]
y = df[TARGET]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create and train One-vs-Rest model
base_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

ovr_model = OneVsRestClassifier(
    estimator=base_model
)

ovr_model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = ovr_model.predict(X_test_scaled)

print("Number of OvR classifiers:", len(ovr_model.estimators_))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=CLASS_ORDER),
        index=[f"Actual {c}" for c in CLASS_ORDER],
        columns=[f"Predicted {c}" for c in CLASS_ORDER]
    )
)

# 7. Predict one new movie
new_movie = pd.DataFrame({
    "IMDb_Rating": [8.5],
    "Audience_Score": [90],
    "Story_Score": [85],
    "Production_Score": [88]
})

new_movie_scaled = scaler.transform(new_movie)
prediction = ovr_model.predict(new_movie_scaled)
probabilities = ovr_model.predict_proba(new_movie_scaled)

print("\nNew Movie Prediction:", prediction[0])
print("\nClass Probabilities:")
print(pd.DataFrame(probabilities, columns=ovr_model.classes_))
