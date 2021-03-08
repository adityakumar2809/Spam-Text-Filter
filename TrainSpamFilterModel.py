from modules.DataInput import readData
from modules.Vectorization import vectorizeTextData
from modules.TFIDFTransformation import tfidfTransformData
from modules.Classification import classify
from modules.ClassificationReport import getClassificationReport,\
                                         getConfusionMatrix

import pickle


def saveModel(model):
    '''Save the model'''
    filename = 'data/TrainedModel.pkl'
    pickle.dump(model, open(filename, 'wb'))


def saveVectorizer(vectorizer):
    '''Save the vectorizer'''
    filename = 'data/Vectorizer.pkl'
    pickle.dump(vectorizer, open(filename, 'wb'))


def saveTransformer(transformer):
    '''Save the transformer'''
    filename = 'data/Transformer.pkl'
    pickle.dump(transformer, open(filename, 'wb'))


def trainSpamFilterModel():
    '''Train the model on given dataset'''
    df = readData(path='data/SMSSpamCollection')
    vectorizer, vectorized_data = vectorizeTextData(df['message'])
    tfidf_transformer, tfidf_transform_data = tfidfTransformData(
                                                vectorized_data
                                              )
    classifier, y_test, y_pred = classify(tfidf_transform_data, df)
    
    saveModel(classifier)
    saveVectorizer(vectorizer)
    saveTransformer(tfidf_transformer)
    
    report = getClassificationReport(y_test, y_pred)
    matrix = getConfusionMatrix(y_test, y_pred)

    print('\n\n======RESULTS======\n\n')
    print('Classification Report:\n')
    print(report)
    print('\n\nConfusion Matrix: \n')
    print(matrix)


def main():
    trainSpamFilterModel()


if __name__ == '__main__':
    main()
