from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def classify(transformed_data, df):
    '''Build an SVC Classifier and predict results'''
    x_train, x_test, y_train, y_test = train_test_split(
                                                         transformed_data,
                                                         df['label'],
                                                         test_size=0.3,
                                                         random_state=50
                                                        )

    classifier = SVC(kernel='linear')
    classifier_fit = classifier.fit(x_train, y_train)

    y_pred = classifier_fit.predict(x_test)
    return classifier_fit, y_test, y_pred
