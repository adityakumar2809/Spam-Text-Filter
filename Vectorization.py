from sklearn.feature_extraction.text import CountVectorizer
from DataPreprocessing import preprocessText

import pandas as pd


def vectorizeTextData(data):
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer = CountVectorizer(analyzer = preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform = vectorizer_fit.transform(data)

    return vectorizer_transform


def main():
    df = pd.read_csv(
                       'SMSSpamCollection',
                       sep='\t',
                       names=["label", "message"]
                    )
    vectorizer_transform = vectorizeTextData(df['message'])
    print(vectorizer_transform)


if __name__ == '__main__':
    main()