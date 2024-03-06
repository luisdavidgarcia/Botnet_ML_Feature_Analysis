import pandas as pd


df = pd.read_csv('combined.csv')
df = df.drop_duplicates(keep='first')
df = df.dropna()
df = df.drop(['Timestamp'], axis=1)

# Determine how percentage of benign to other traffice in Label column
print(df['Label'].value_counts())