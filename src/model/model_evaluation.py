import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

try:
    # Fetch the data from data/features
    logger.info("Loading test data from './data/features/test_bow.csv'.")
    test_processed_data = pd.read_csv('./data/features/test_bow.csv')
    logger.debug("Test data loaded successfully.")
except FileNotFoundError:
    logger.error("The specified test data file was not found in './data/features'.")
    raise
except Exception as e:
    logger.error(f"Failed to load test data: {e}")
    raise

try:
    # Prepare test data
    logger.info("Preparing test data for evaluation.")
    X_test_bow = test_processed_data.iloc[:, :-1].values
    y_test = test_processed_data.iloc[:, -1].values
    logger.debug("Test data prepared successfully.")
except Exception as e:
    logger.error(f"Error while preparing test data: {e}")
    raise

try:
    # Load the trained model
    model_path = './models/model.pkl'
    logger.info(f"Loading the trained model from '{model_path}'.")
    with open(model_path, 'rb') as model_file:
        xgb_model = pickle.load(model_file)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error(f"The trained model file '{model_path}' was not found.")
    raise
except Exception as e:
    logger.error(f"Error while loading the model: {e}")
    raise

try:
    # Make predictions
    logger.info("Making predictions using the trained model.")
    y_pred = xgb_model.predict(X_test_bow)
    y_pred_proba = xgb_model.predict_proba(X_test_bow)[:, 1]
    logger.debug("Predictions made successfully.")
except Exception as e:
    logger.error(f"Error during model predictions: {e}")
    raise

try:
    # Calculate evaluation metrics
    logger.info("Calculating evaluation metrics.")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    logger.debug(f"Evaluation metrics: {metrics_dict}")
except Exception as e:
    logger.error(f"Error while calculating evaluation metrics: {e}")
    raise

try:
    # Define the path to save the metrics
    metrics_path = 'reports/metrics.json'  # Save in the 'reports' folder
    logger.info(f"Saving evaluation metrics to '{metrics_path}'.")

    # Save the dictionary to the 'reports/metrics.json' file
    with open(metrics_path, "w") as json_file:
        json.dump(metrics_dict, json_file, indent=4)
    
    logger.info("Evaluation metrics saved successfully.")

except PermissionError:
    logger.error("Permission denied when saving the metrics file 'metrics.json'.")
    raise
except Exception as e:
    logger.error(f"Failed to save evaluation metrics: {e}")
    raise
