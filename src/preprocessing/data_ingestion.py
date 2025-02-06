import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.debug(f'Test size retrieved: {test_size}')
            if not isinstance(test_size, float):
                logger.error("Test size should be a float.")
                raise ValueError("Test size should be a float.")
        return test_size
    except FileNotFoundError:
        logger.error(f"The file '{params_path}' was not found.")
        raise
    except KeyError as e:
        logger.error(f"Missing key {e} in 'params.yaml'.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse the YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in load_params: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.debug(f"Data successfully read from URL: {url}")
        return df
    except Exception as e:
        logger.error(f"Failed to read the data from URL '{url}': {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'tweet_id' not in df.columns or 'sentiment' not in df.columns:
            logger.error("Required columns ('tweet_id' and 'sentiment') are missing from the DataFrame.")
            raise KeyError("Required columns ('tweet_id' and 'sentiment') are missing from the DataFrame.")
        
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        
        if final_df.empty:
            logger.error("No rows found with 'happiness' or 'sadness' in the 'sentiment' column.")
            raise ValueError("No rows found with 'happiness' or 'sadness' in the 'sentiment' column.")
        
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.debug("Data processing completed successfully.")
        return final_df
    except KeyError as e:
        logger.error(f"KeyError in process_data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_data: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug(f"Data successfully saved to {data_path}")
    except PermissionError:
        logger.error(f"Permission denied when creating or saving to '{data_path}'.")
        raise
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise


try:
    params_path = 'params.yaml'
    logger.info("Starting data ingestion process.")
    
    test_size = load_params(params_path)
    logger.info(f"Loaded test size: {test_size}")
    
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    logger.info("Data successfully read from URL.")
    
    final_df = process_data(df)
    logger.info("Data processing completed.")
    
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    logger.info("Data split into training and test sets.")
    
    data_path = os.path.join("data","raw")
    save_data(data_path, train_data, test_data)
    logger.info("Data saved successfully.")
    
    print("Data processing and saving completed successfully.")
except Exception as e:
    logger.error(f"Error in main execution: {e}")
    print(f"Error in main execution: {e}")


