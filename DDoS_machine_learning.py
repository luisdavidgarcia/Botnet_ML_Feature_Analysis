import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
day15 = pd.read_csv('/content/drive/MyDrive/DDOS_Datasets/CSE-CIC-IDS2018/02-15-2018.csv')
main_df = day15.drop_duplicates(keep='first')
one_value = main_df.columns[main_df.nunique() == 1]
main_df_2 = main_df.drop(columns=one_value, axis=1)

# Drop unnecessary columns
columns_to_drop = ['Timestamp']
for col in columns_to_drop:
    if col in main_df_2.columns:
        main_df_2 = main_df_2.drop(col, axis=1)

# Clean data, remove too big of values
main_df_2 = main_df_2.replace([np.inf, -np.inf], np.nan).dropna()

# Handle the target variable
if main_df_2['Label'].dtype == 'object':
    le = LabelEncoder()
    main_df_2['Label'] = le.fit_transform(main_df_2['Label'])


# Separating features and target
X = main_df_2.drop('Label', axis=1)
y = main_df_2['Label']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=1000)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train and evaluate each model
models = [rf_model, dt_model, lr_model, xgb_model]
model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'XGBoost']

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Continue with your RFE steps...
