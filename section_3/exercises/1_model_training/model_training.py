import os
import pandas as pd
import numpy as np
import mlflow
import wandb
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Initialize WandB for tracking the model training process
wandb.init(project="credit-card-fraud-detection-test-515", name="model-training")

def load_data():
    """
    Load the engineered dataset from WandB Artifacts.
    
    :return: pandas DataFrame, feature matrix and target vector
    """
    artifact = wandb.use_artifact('anicin/credit-card-fraud-detection-test-515/engineered_data.csv:latest', type='dataset')
    artifact_dir = artifact.download()

    # Load the dataset
    data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
    
    # Assume the target variable is 'is_fraud' and features are the rest
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model and evaluate its performance.
    
    :param X_train: Training feature matrix
    :param y_train: Training target vector
    :param X_test: Test feature matrix
    :param y_test: Test target vector
    :return: trained model, evaluation metrics
    """
    # TODO: Initialize and train the XGBoost model with the eval_metric='logloss' parameter
    # TODO: Use use_label_encoder=False to avoid warnings  and train the model

    # TODO: Make predictions on the test set
    
    # TODO:  Calculate evaluation metrics (accuracy, precision, recall, f1_score) and store them in the metrics dictionary

    # Log metrics to WandB
    wandb.log(metrics)

    # TODO: Infer and return signature for the model deployment

    return model, metrics, signature

def main():
    mlflow.start_run()

    # Load the data from WandB
    X, y = load_data()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and evaluate its performance
    model, metrics, signature = train_model(X_train, y_train, X_test, y_test)

    # TODO: Log parameters and metrics to MLFlow
    
    # NOTE (HINT): To get metrics you can use metrics.items() -> returns dictionary

    # TODO: Save the model to a file
    
    # TODO: Log the model to WandB and MLFlow

    # TODO: Log model to MLFlow and prepare it for deployment
    # NOTE (HINT): Use mlflow.xgboost.log_model()

    mlflow.end_run()

if __name__ == "__main__":
    main()
