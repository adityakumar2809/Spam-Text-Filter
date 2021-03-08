from modules.DataInput import readData
from modules.Vectorization import vectorizeTextData
from modules.TFIDFTransformation import tfidfTransformData
from modules.Classification import classify
from modules.ClassificationReport import getClassificationReport, getConfusionMatrix

import pickle


def saveModel(model):
    filename = 'TrainedModel.sav'
    pickle.dump(model, open(filename, 'wb'))


def trainSpamFilterModel():
    df = readData(path='data/SMSSpamCollection')
    vectorized_data = vectorizeTextData(df)
    tfidf_transform_data = tfidfTransformData(vectorized_data)
    classifier, y_test, y_pred = classify(tfidf_transform_data, df)
    saveModel(classifier)
    report = getClassificationReport(y_test, y_pred)
    matrix = getConfusionMatrix(y_test, y_pred)

    print('\n\n======RESULTS======\n\n')
    print('Classification Report:\n')
    print(report)
    print('\n\nConfusion Matrix: \n')
    print(matrix)


def main():
    filterSpamMessages()


if __name__ == '__main__':
    main()
