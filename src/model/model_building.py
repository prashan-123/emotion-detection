import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import yaml
import logging

# Configure logging
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

try:
    # Load parameters from params.yaml
    logger.info("Loading parameters from 'params.yaml'.")
    params = yaml.safe_load(open('params.yaml', 'r'))['model_building']
    logger.debug(f"Model parameters retrieved: {params}")
except FileNotFoundError:
    logger.error("The file 'params.yaml' was not found.")
    raise
except KeyError as e:
    logger.error(f"Missing key '{e}' in 'params.yaml'.")
    raise
except yaml.YAMLError as e:
    logger.error(f"Failed to parse 'params.yaml': {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error while loading parameters: {e}")
    raise

try:
    # Fetch the data from data/features
    logger.info("Loading feature-engineered data from './data/features'.")
    train_processed_data = pd.read_csv('./data/features/train_tfidf.csv')
    logger.debug("Feature-engineered data loaded successfully.")
except FileNotFoundError:
    logger.error("The specified feature-engineered data file was not found in './data/features'.")
    raise
except Exception as e:
    logger.error(f"Failed to load feature-engineered data: {e}")
    raise

try:
    # Prepare data for model training
    logger.info("Preparing data for model training.")
    X_train_bow = train_processed_data.iloc[:, 0:-1].values
    y_train = train_processed_data.iloc[:, -1].values
    logger.debug("Data prepared for model training.")
except Exception as e:
    logger.error(f"Error while preparing data for model training: {e}")
    raise

try:
    # Define and train the XGBoost model
    logger.info("Training the XGBoost model.")
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        eta=params['eta'],
        gamma=params['gamma']
    )
    xgb_model.fit(X_train_bow, y_train)
    logger.info("XGBoost model trained successfully.")
except Exception as e:
    logger.error(f"Error during model training: {e}")
    raise

try:
    # Save the model
    model_path = './models/model.pkl'  # Ensure correct absolute path
    logger.info(f"Saving the trained model to '{model_path}'.")
    with open(model_path, 'wb') as model_file:
        pickle.dump(xgb_model, model_file)
    logger.info("Model saved successfully.")
except PermissionError:
    logger.error(f"Permission denied when saving the model to '{model_path}'.")
    raise
except Exception as e:
    logger.error(f"Failed to save the model: {e}")
    raise