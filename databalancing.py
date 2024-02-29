import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

def prepare_balanced_data(df, random_state=42, test_size=0.25):
    # Drop rows with NaN values across the entire DataFrame to ensure alignment
    df_cleaned = df.dropna()

    # Select features excluding 'object' types and the label
    x = df_cleaned.select_dtypes(exclude=['object'])  
    y = df_cleaned['Label'].replace(['DoS attacks-GoldenEye', 'DoS attacks-Slowloris'], 'DoS')

    # Replace infinity values with NaN in X and then drop any rows with NaN values to clean both X and y together
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.dropna(inplace=True)

    # Ensuring y is aligned with X after dropping rows, before encoding
    y = y.loc[x.index]

    # Encoding the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Now, X and y_encoded are aligned and ready for train-test split and further processing

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test

def apply_smote(x_train, y_train, random_state=42, sampling_strategy='auto'):
    # Applying SMOTE
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

    # Checking the class distribution
    print(f"Original class distribution: {Counter(y_train)}")
    print(f"Resampled class distribution: {Counter(y_resampled)}")

    return x_resampled, y_resampled

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("02-15-2018_1.csv")

    # Prepare balanced data
    x_train, x_test, y_train, y_test = prepare_balanced_data(df)
    x_resampled, y_resampled = apply_smote(x_train, y_train, sampling_strategy=0.45)