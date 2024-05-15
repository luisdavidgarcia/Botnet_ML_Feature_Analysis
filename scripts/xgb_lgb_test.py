import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

# Assuming prepare_norm_balanced_data is a function from 'databalancing'
# that normalizes and balances the dataset and returns train-test splits and label map.
from databalancing import prepare_norm_balanced_data

# Load the dataset
path = "/Users/lucky/GitHub/BotnetFeatureSelection/data/cleaned_combined.csv"
df = pd.read_csv(path)
x_train, x_test, y_train, y_test, label_map = prepare_norm_balanced_data(df)

# Function to train and evaluate a model
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    predictions = model.predict(x_test)
    prediction_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(classification_report(y_test, predictions))

# Train and evaluate LightGBM
print("Evaluating LightGBM:")
lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
train_and_evaluate(lgb_model, x_train, y_train, x_test, y_test)

# Train and evaluate XGBoost
print("Evaluating XGBoost:")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
train_and_evaluate(xgb_model, x_train, y_train, x_test, y_test)
