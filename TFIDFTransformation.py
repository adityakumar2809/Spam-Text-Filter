from sklearn.feature_extraction.text import TfidfTransformer
from DataInput import readData
from Vectorization import vectorizeTextData


def tfidfTransformData(vectorized_data):
    '''Calculate Term Frequency And Inverse Document Frequency'''
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer_fit = tfidf_transformer.fit(vectorized_data)
    tfidf_transformed_data = tfidf_transformer_fit.transform(vectorized_data)

    return tfidf_transformed_data


def main():
    df = readData(path='SMSSpamCollection')
    vectorized_data = vectorizeTextData(df['message'])
    tfidf_transformed_data = tfidfTransformData(vectorized_data)
    print(tfidf_transformed_data.shape)


if __name__ == '__main__':
    main()