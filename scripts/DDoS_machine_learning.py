import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from databalancing import prepare_balanced_data, apply_smote

# Load the dataset
df = pd.read_csv('02-15-2018.csv')
x_train, x_test, y_train, y_test, label_map = prepare_balanced_data(df)

# Benign Ratio Ranges
benign_ratios = [0.5, 0.7, 0.8]

# Apply SMOTE
for benign_ratio in benign_ratios:
    print("---------------------------------------------------------------------")
    print(f"Applying SMOTE with benign ratio: {benign_ratio}")
    x_resampled, y_resampled = apply_smote(x_train, y_train, begnin_ratio=benign_ratio)

    print(f"Len of x_train: {len(x_train)} and len of x_resampled: {len(x_resampled)}")
    print("---------------------------------------------------------------------")

    # Initialize models
    rf_model = RandomForestClassifier(random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    lr_model = LogisticRegression(max_iter=1000)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train and evaluate each model
    models = [rf_model, dt_model, lr_model, xgb_model]
    model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'XGBoost']

    for model, name in zip(models, model_names):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    print("---------------------------------------------------------------------")
    # Continue with your RFE steps...
