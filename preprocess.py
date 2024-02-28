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

# Function to check for multiple data types in a column
def check_column_types(dataframe):
    for column in dataframe.columns:
        unique_types = {type(v).__name__ for v in dataframe[column]}
        if len(unique_types) > 1:
            print(f"Column '{column}' has multiple types: {unique_types}")
        else:
            print(f"Column '{column}' has a single type: {unique_types.pop()}")

# Check each column in the DataFrame
check_column_types(main_df_2)
