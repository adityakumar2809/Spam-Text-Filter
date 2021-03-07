from sklearn.feature_extraction.text import CountVectorizer
from DataPreprocessing import preprocessText

import pandas as pd


def vectorizeTextData(data): # data corresponds to df['message']
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer = CountVectorizer(analyzer = preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform = vectorizer_fit.transform(data)

    return vectorizer_transform
