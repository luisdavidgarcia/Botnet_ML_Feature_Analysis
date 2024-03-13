import pandas as pd

def main():
    df= pd.read_csv('data/cleaned_combined.csv')
    # remove minority attacks of "DDOS attack-LOIC-UDP"
    df = df[df['Label'] != 'DDOS attack-LOIC-UDP']
    
    # print how many labels are in the label column now
    print(df['Label'].value_counts())

    # save the new dataset to a new file
    df.to_csv('data/combined_no_LOIC_UDP.csv', index=False)

if __name__ == '__main__':
    main()
