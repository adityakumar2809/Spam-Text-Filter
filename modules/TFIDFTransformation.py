from sklearn.feature_extraction.text import TfidfTransformer


def tfidfTransformData(vectorized_data):
    '''Calculate Term Frequency And Inverse Document Frequency'''
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer_fit = tfidf_transformer.fit(vectorized_data)
    tfidf_transformed_data = tfidf_transformer_fit.transform(vectorized_data)

    return tfidf_transformer_fit, tfidf_transformed_data
