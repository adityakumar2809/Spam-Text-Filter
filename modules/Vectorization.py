from sklearn.feature_extraction.text import CountVectorizer
from .DataPreprocessing import preprocessText


def vectorizeTextData(data):
    '''Vectorize text DataFrame using CountVectorizer'''
    vectorizer = CountVectorizer(analyzer=preprocessText)
    vectorizer_fit = vectorizer.fit(data)
    vectorizer_transform_data = vectorizer_fit.transform(data)

    return vectorizer_fit, vectorizer_transform_data
