import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the dataset
df = pd.read_csv("02-15-2018_1.csv")

# Drop rows with NaN values across the entire DataFrame to ensure alignment
df_cleaned = df.dropna()

# Select features excluding 'object' types and the label
X = df_cleaned.select_dtypes(exclude=['object'])  # Features
y = df_cleaned['Label']  # Assuming 'Label' is the name of your target column

# At this point, X and y are aligned and have the same number of rows

# Replace infinity values with NaN in X and then drop any rows with NaN values to clean both X and y together
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# Ensuring y is aligned with X after dropping rows, before encoding
y = y.loc[X.index]

# Encoding the labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Now, X and y_encoded are aligned and ready for train-test split and further processing

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Checking the class distribution
print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_resampled)}")
