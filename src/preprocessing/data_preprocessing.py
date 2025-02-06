import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Download NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

# Fetch the data from data/raw
try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.info("Data successfully loaded from './data/raw'.")
except FileNotFoundError:
    logger.error("The specified data files were not found in './data/raw'.")
    raise
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# Text preprocessing functions
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        raise

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in remove_stop_words: {e}")
        raise

def removing_numbers(text):
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error(f"Error in removing_numbers: {e}")
        raise

def lower_case(text):
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in lower_case: {e}")
        raise

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"Error in removing_punctuations: {e}")
        raise

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error in removing_urls: {e}")
        raise

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logger.error(f"Error in remove_small_sentences: {e}")
        raise

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lambda content: lower_case(content))
        df['content'] = df['content'].apply(lambda content: remove_stop_words(content))
        df['content'] = df['content'].apply(lambda content: removing_numbers(content))
        df['content'] = df['content'].apply(lambda content: removing_punctuations(content))
        df['content'] = df['content'].apply(lambda content: removing_urls(content))
        df['content'] = df['content'].apply(lambda content: lemmatization(content))
        logger.info("Text normalization completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        raise

# Process the data
try:
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    logger.info("Data preprocessing completed successfully.")
except Exception as e:
    logger.error(f"Error during data preprocessing: {e}")
    raise

# Store the data inside data/processed
try:
    data_path = os.path.join("data", "processed")
    os.makedirs(data_path, exist_ok=True)
    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    logger.info(f"Processed data saved successfully to {data_path}.")
except PermissionError:
    logger.error(f"Permission denied when creating or saving to '{data_path}'.")
    raise
except Exception as e:
    logger.error(f"Failed to save processed data: {e}")
    raise