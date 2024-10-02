# Import necessary libraries
import mlflow  # MLFlow for experiment tracking and model management
import mlflow.sklearn  # MLFlow's Scikit-learn integration
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
print(f"MLFlow Version: {mlflow.__version__}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
# Step 1: Initialize MLFlow
# Mention that MLFlow is particularly strong in model versioning and deployment.
mlflow.set_experiment("mlops-intro")

# Step 2: Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Step 4: Train the model and log it with MLFlow

with NotImplementedError:
    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # TODO: Log the accuracy with MLFlow

    # TODO: Log the model with MLFlow

    # TODO: Save the model with joblib and log it as an artifact with MLFlow


    # TODO: Log model parameters for reproducibility

# Step 5: Finish WandB and MLFlow runs
# TODO: Ensure both MLFlow and WandB runs are properly closed.
