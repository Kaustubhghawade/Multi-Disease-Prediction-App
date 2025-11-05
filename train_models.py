import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Ensure 'models' folder exists
if not os.path.exists("models"):
    os.makedirs("models")

def train_and_save_model(dataset_path, target_column, model_name):
    """
    Train a Logistic Regression model and save it with scaler and features.
    """
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Keep only numeric columns (drop non-numeric, e.g., 'name' column in Parkinson's)
    data = data.select_dtypes(include=['float64', 'int64'])

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"{model_name} Model Accuracy: {accuracy*100:.2f}%")

    # Save model, scaler, and feature names
    with open(f"models/{model_name}_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "features": X.columns.tolist()
        }, f)

# Train models for all three diseases
train_and_save_model("datasets/diabetes.csv", "Outcome", "diabetes")
train_and_save_model("datasets/heart.csv", "target", "heart")
train_and_save_model("datasets/parkinsons.csv", "status", "parkinsons")
