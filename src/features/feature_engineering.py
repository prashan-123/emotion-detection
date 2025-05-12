import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Configure logging
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

try:
    # Load parameters from params.yaml
    logger.info("Loading parameters from 'params.yaml'.")
    params = yaml.safe_load(open('params.yaml', 'r'))
    max_features = params['feature_engineering']['max_features']
    logger.debug(f"Max features retrieved: {max_features}")
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
    # Fetch the data from data/processed
    logger.info("Loading processed data from './data/processed'.")
    train_processed_data = pd.read_csv('./data/processed/train_processed.csv')
    test_processed_data = pd.read_csv('./data/processed/test_processed.csv')
    logger.debug("Processed data loaded successfully.")
except FileNotFoundError:
    logger.error("The specified processed data files were not found in './data/processed'.")
    raise
except Exception as e:
    logger.error(f"Failed to load processed data: {e}")
    raise

try:
    # Handle missing values
    logger.info("Handling missing values in the dataset.")
    train_processed_data.fillna('', inplace=True)
    test_processed_data.fillna('', inplace=True)
    logger.debug("Missing values handled successfully.")
except Exception as e:
    logger.error(f"Error while handling missing values: {e}")
    raise

try:
    # Prepare data for Bag of Words (BoW)
    logger.info("Preparing data for Bag of Words (BoW) transformation.")
    X_train = train_processed_data['content'].values
    y_train = train_processed_data['sentiment'].values
    X_test = test_processed_data['content'].values
    y_test = test_processed_data['sentiment'].values
    logger.debug("Data prepared for BoW transformation.")
except KeyError as e:
    logger.error(f"Missing required column in the dataset: {e}")
    raise
except Exception as e:
    logger.error(f"Error while preparing data for BoW: {e}")
    raise

try:
    # Apply Tfidf (TfidfVectorizer)
    logger.info("Applying Tfidf (TfidfVectorizer) transformation.")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    logger.debug("Tfidf transformation completed successfully.")
except Exception as e:
    logger.error(f"Tfidf during Tfidf transformation: {e}")
    raise

try:
    # Create DataFrames from BoW results
    logger.info("Creating DataFrames from BoW results.")
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test
    logger.debug("DataFrames created successfully.")
except Exception as e:
    logger.error(f"Error while creating DataFrames: {e}")
    raise

try:
    # Store the data inside data/features
    logger.info("Saving feature-engineered data to './data/features'.")
    data_path = os.path.join("data", "features")
    os.makedirs(data_path, exist_ok=True)
    train_df.to_csv(os.path.join(data_path, "train_tfidf.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_tfidf.csv"), index=False)
    logger.info("Feature-engineered data saved successfully.")
except PermissionError:
    logger.error(f"Permission denied when creating or saving to '{data_path}'.")
    raise
except Exception as e:
    logger.error(f"Failed to save feature-engineered data: {e}")
    raise