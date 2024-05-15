import pandas as pd

def main():
    # Load the dataset
    df = pd.read_csv('/Users/lucky/GitHub/BotnetFeatureSelection/data/cleaned_combined.csv')

    # Identify columns to keep: those not of object type or the 'Label' column
    columns_to_keep = df.select_dtypes(exclude=['object']).columns.tolist() + ['Label']
    
    # Drop columns not in 'columns_to_keep'
    df = df[columns_to_keep]

    # Remove specific label entries
    df = df[df['Label'] != 'DDOS attack-LOIC-UDP']

    # Features to drop:
    features_to_drop = [
        "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", 
        "CWE Flag Count", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", 
        "Bwd Blk Rate Avg" 
    ]

    df = df.drop(columns=features_to_drop, axis=1)
    # df = df.drop_duplicates().dropna()

    # Output dataframe info
    print(df.head())
    print(df.shape)
    print(df['Label'].value_counts())

    # Calculate the ratio of each label
    for label in df['Label'].unique():
        print(f"{label}: {df[df['Label'] == label].shape[0] / df.shape[0] * 100:.2f}%")

    # Save cleaned DataFrame to new file (uncomment to use)
    # df.to_csv('data/combined_no_LOIC_UDP.csv', index=False)

if __name__ == '__main__':
    main()
