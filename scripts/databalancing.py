import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

def prepare_norm_balanced_data(df, top_features, random_state=42, test_size=0.25, remove_duplicates=True, modify_inplace=True):
    """ Prepares the data for training and testing.

    Args:
        df: pandas DataFrame.
        top_features: List of top features to select.
        random_state: int, default=42.
        test_size: float, default=0.25.
        remove_duplicates: bool, default=True.
        modify_inplace: bool, default=False. If True, modifies the DataFrame in-place.

    Returns:
        x_train_scaled: pandas DataFrame (scaled training data).
        x_test_scaled: pandas DataFrame (scaled testing data).
        y_train: pandas Series (training labels).
        y_test: pandas Series (testing labels).
        label_mapping: dict (label encoding mapping).
    """

    if not modify_inplace:
        df = df.copy()  # Work on a copy if in-place modification is not desired

    # Data Cleaning
    if remove_duplicates:
        df.drop_duplicates(keep='first', inplace=True)

    df.dropna(subset=['Label'], inplace=True) 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Columns to Remove 
    cols_to_drop = ['Timestamp'] + df.columns[df.nunique() == 1].tolist()
    df.drop(columns=cols_to_drop, inplace=True) 

    # Label Encoding
    if df['Label'].dtype == 'object':
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])
        label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    else:
        label_mapping = None

    # Feature Selection and Data Split
    X = df[top_features]
    y = df['Label']

    # Normalization and Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index), \
           pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index), \
           y_train, y_test, label_mapping


def apply_smote(x_train, y_train, random_state=42, benign_ratio=0.0):
    """ This function applies SMOTE to the training data to balance the classes.

    Args:
        x_train: pandas DataFrame
        y_train: pandas Series
        random_state: int, default=42
        benign_ratio=: float, default=0.0

    Returns:
        x_resampled: pandas DataFrame
        y_resampled: pandas Series
    """

    if benign_ratio == 0.0:
        smote = SMOTE(random_state=random_state)
        x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

        # Checking the class distribution
        print(f"Total samples in the training set: {len(y_train)}")
        print(f"Original class distribution: {Counter(y_train)}")
        print(f"Resampled class distribution: {Counter(y_resampled)}")
        return x_resampled, y_resampled

    # Calculate the desired number of samples for each class
    count_labels = Counter(y_train)
    
    dos_labels_sum = count_labels[1] + count_labels[2]
    goldeneye_ratio = count_labels[1] / dos_labels_sum
    slowloris_ratio = count_labels[2] / dos_labels_sum

    total_benign = count_labels[0]
    total_samples = len(y_train)
    total_samples_to_add_to_dos =  (total_benign/benign_ratio - total_samples)
    desired_samples_goldeneye = int(total_samples_to_add_to_dos * goldeneye_ratio) + count_labels[1]
    desired_samples_slowloris = int(total_samples_to_add_to_dos * slowloris_ratio) + count_labels[2]

    # Applying SMOTE
    smote = SMOTE(random_state=random_state, sampling_strategy={1: desired_samples_goldeneye, 2: desired_samples_slowloris})
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

    # Checking the class distribution
    print(f"Total samples in the training set: {len(y_train)}")
    print(f"Original class distribution: {Counter(y_train)}")
    print(f"Total samples in the resampled set: {len(y_resampled)}")
    print(f"Resampled class distribution: {Counter(y_resampled)}")

    return x_resampled, y_resampled
