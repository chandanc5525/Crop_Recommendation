import pandas as pd
import logging
from sklearn.model_selection import train_test_split

def load_data(url):
    logging.info("Loading data from URL")
    return pd.read_csv(url)

def split_data(df, test_size, random_state):
    X = df.drop('label', axis=1)
    y = df['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
