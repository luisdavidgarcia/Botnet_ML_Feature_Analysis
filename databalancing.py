import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

def prepare_balanced_data(df, random_state=42, test_size=0.25):
    # Drop rows with NaN values across the entire DataFrame to ensure alignment
    # df_cleaned = df.dropna()

    main_df = df.drop_duplicates(keep='first')
    one_value = main_df.columns[main_df.nunique() == 1]
    main_df_2 = main_df.drop(columns=one_value, axis=1)

    # Drop unnecessary columns
    columns_to_drop = ['Timestamp']
    for col in columns_to_drop:
        if col in main_df_2.columns:
            main_df_2 = main_df_2.drop(col, axis=1)

    # Clean data, remove too big of values
    main_df_2 = main_df_2.replace([np.inf, -np.inf], np.nan).dropna()

    # Create a label mapping
    label_mapping = {}

    # Handle the target variable
    if main_df_2['Label'].dtype == 'object':
        le = LabelEncoder()
        main_df_2['Label'] = le.fit_transform(main_df_2['Label'])
        # Mapping the labels to the encoded values
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Select features excluding 'object' types and the label
    x = main_df_2.drop('Label', axis=1)
    y = main_df_2['Label']

    print("Label mapping:", label_mapping)
    print("Total counts per label in y:", y.value_counts())
    print("Total number of rows in y:", y.shape[0])
    total_goldeneye = y.value_counts()[label_mapping['DoS attacks-GoldenEye']]
    total_slowloris = y.value_counts()[label_mapping['DoS attacks-Slowloris']]
    total_benign = y.value_counts()[label_mapping['Benign']]    
    print(f"Percentage of 'DoS attacks-GoldenEye' in y: {total_goldeneye/y.shape[0]*100:.2f}%")
    print(f"Percentage of 'DoS attacks-Slowloris' in y: {total_slowloris/y.shape[0]*100:.2f}%")
    print(f"Percentage of 'Benign' in y: {total_benign/y.shape[0]*100:.2f}%")

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

    return x_train, x_test, y_train, y_test, label_mapping

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
    df = pd.read_csv("02-15-2018.csv")

    # Prepare balanced data
    x_train, x_test, y_train, y_test, label_map = prepare_balanced_data(df)
    # x_resampled, y_resampled = apply_smote(x_train, y_train, sampling_strategy={1: 740969, 2: 740969})