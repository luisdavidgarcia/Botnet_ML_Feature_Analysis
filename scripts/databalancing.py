import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

def prepare_norm_balanced_data(df, top_features, random_state=42, test_size=0.25):
    """ This function prepares the data for training and testing by removing duplicates, unnecessary columns, and cleaning the data.
    Args:
        df: pandas DataFrame 
        random_state: int, default=42
        test_size: float, default=0.25

    Returns:
        x_train_scaled: pandas DataFrame that has been scaled
        x_test_scaled: pandas DataFrame that has been scaled
        y_train: pandas Series
        y_test: pandas Series
        label_mapping: dict
    """

    main_df = df.drop_duplicates(keep='first')
    one_value = main_df.columns[main_df.nunique() == 1]
    main_df_2 = main_df.drop(columns=one_value, axis=1)
    columns_to_drop = ['Timestamp']
    for col in columns_to_drop:
        if col in main_df_2.columns:
            main_df_2 = main_df_2.drop(col, axis=1)

    main_df_2 = main_df_2.replace([np.inf, -np.inf], np.nan).dropna()
    label_mapping = {}
    if main_df_2['Label'].dtype == 'object':
        le = LabelEncoder()
        main_df_2['Label'] = le.fit_transform(main_df_2['Label'])
        label_mapping = {index: label for index, label in enumerate(le.classes_)}

    x = main_df_2.drop('Label', axis=1)
    y = main_df_2['Label']

    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.dropna(inplace=True)
    x = x[top_features]
    y = y.loc[x.index]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    x_train_scaled, x_test_scaled, y_train, y_test = normalize_data_and_split(x, y_encoded, random_state=random_state, test_size=test_size)

    print(f"Label mapping: {label_mapping}")
    label_counts = Counter(y_train)
    print(f"Label distribution in the training set: {label_counts}")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count / len(y_train) * 100:.2f}%, {count} samples")
    print(f"Total samples in the training set: {len(y_train)}")

    return x_train_scaled, x_test_scaled, y_train, y_test, label_mapping

def normalize_data_and_split(x, y, random_state=42, test_size=0.25):
    """ This function normalizes the data and splits it into training and testing sets.

    Args:
        x: pandas DataFrame
        y: pandas Series
        random_state: int, default=42
        test_size: float, default=0.25

    Returns:
        x_train_scaled: pandas DataFrame
        x_test_scaled: pandas DataFrame
        y_train: pandas Series
        y_test: pandas Series
    """

    # Initialize the scaler
    scaler = StandardScaler()

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Fit on training set only
    x_train_scaled = scaler.fit_transform(x_train)
    
    # Apply transform to both the training set and the test set
    x_test_scaled = scaler.transform(x_test)

    # Convert arrays back to DataFrame for convenience
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)
    
    return x_train_scaled, x_test_scaled, y_train, y_test

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
