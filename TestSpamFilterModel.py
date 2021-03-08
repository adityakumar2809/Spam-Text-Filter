from modules.Vectorization import vectorizeTextData
from modules.TFIDFTransformation import tfidfTransformData

import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def getTestData():
    '''Take input text from user'''
    input_text = input('Enter messages to be verified separated by \';\'\n')
    text_list = input_text.split(';')
    return text_list


def loadSavedModel(path='TrainedModel.sav'):
    '''Load the saved trained model for use'''
    loaded_model = pickle.load(open(path, 'rb'))
    return loaded_model


def predictResult(model, input_text):
    '''Predict the result for the given data'''
    vectorized_data = vectorizeTextData(input_text)
    x_test = tfidfTransformData(vectorized_data)
    
    ## APPLY RESHAPING OF TESTING DATA HERE
    
    y_pred = model.predict(x_test)
    return y_pred


def printResult(x_test, y_pred):
    for i, text in enumerate(x_test):
        print(textwrap.shorten(text, width=20, placeholder='...'), '\t', y_pred[i])


def main():
    input_text = getTestData()
    classifier = loadSavedModel(path='TrainedModel.sav')
    y_pred = predictResult(classifier, input_text)
    printResult(input_text, y_pred)


if __name__ == '__main__':
    main()