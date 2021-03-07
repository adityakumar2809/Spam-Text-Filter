import pandas as pd


def readData(path='Data/SMSSpamCollection'):
    '''Read tab separated CSV and return the dataframe'''
    df = pd.read_csv(
                       path,
                       sep='\t',
                       names=["label", "message"]
                    )
    return df


def main():
    path = 'Data/SMSSpamCollection'
    df = readData(path)
    print(df.describe())


if __name__ == '__main__':
    main()