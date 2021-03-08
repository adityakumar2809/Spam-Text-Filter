import pickle
import textwrap


def getTestData():
    '''Take input text from user'''
    input_text = input('Enter messages to be verified separated by \';\'\n')
    text_list = input_text.split(';')
    return text_list


def loadSavedFile(path):
    '''Load the saved data files for use'''
    loaded_data = pickle.load(open(path, 'rb'))
    return loaded_data


def preprocessData(input_text, vectorizer, transformer):
    '''Transform the data using the transformer for training data'''
    vectorized_data = vectorizer.transform(input_text)
    transformed_data = transformer.transform(vectorized_data)
    return transformed_data


def predictResult(test_data, model):
    '''Predict the result for the given data'''
    y_pred = model.predict(test_data)
    return y_pred


def printResult(x_test, y_pred):
    for i, text in enumerate(x_test):
        print(
            textwrap.shorten(text, width=20, placeholder='...'),
            '\t',
            y_pred[i]
        )


def main():
    input_text = getTestData()

    classifier = loadSavedFile(path='data/TrainedModel.pkl')
    vectorizer = loadSavedFile(path='data/Vectorizer.pkl')
    transformer = loadSavedFile(path='data/Transformer.pkl')

    test_data = preprocessData(input_text, vectorizer, transformer)

    predictions = predictResult(test_data, classifier)
    printResult(input_text, predictions)


if __name__ == '__main__':
    main()
