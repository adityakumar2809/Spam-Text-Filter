from sklearn.feature_extraction.text import CountVectorizer
from .DataPreprocessing import preprocessText

import pandas as pd


def vectorizeTextData(df):
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer = CountVectorizer(analyzer = preprocessText)
    vectorizer_fit = vectorizer.fit(df['message'])
    vectorizer_transform = vectorizer_fit.transform(df['message'])

    return vectorizer_transform
