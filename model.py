# Import Data manipulation libraries
import pandas as pd
import numpy as np
# Import Data Visulaztion libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Import machine learning libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Import Logging
import logging
logging.basicConfig(level=logging.INFO,
                    filemode = 'w',
                    filename = 'log.txt',
                    format = '%(asctime)s - %(levelname)s - %(message)s',force =True)
# Import Filterwarning Library
import warnings
warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/chandanc5525/Crop_Recommendation/refs/heads/main/Crop_Recommendation.csv'

df = pd.read_csv(url)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Crop'] = le.fit_transform(df['Crop'])

X = df.drop(columns = ['Crop','Rainfall'], axis = 1)
y = df['Crop']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from  sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))