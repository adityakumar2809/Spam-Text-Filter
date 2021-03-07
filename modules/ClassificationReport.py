from sklearn.metrics import classification_report, confusion_matrix


def getClassificationReport(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    return report


def getConfusionMatrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    return matrix
