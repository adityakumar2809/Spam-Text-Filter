from DataInput import readData
from Vectorization import vectorizeTextData
from TFIDFTransformation import tfidfTransformData
from Classification import classify
from ClassificationReport import getClassificationReport, getConfusionMatrix


def filterSpamMessages():
    df = readData(path='Data/SMSSpamCollection')
    vectorized_data = vectorizeTextData(df)
    tfidf_transform_data = tfidfTransformData(vectorized_data)
    y_test, y_pred = classify(tfidf_transform_data, df)
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
