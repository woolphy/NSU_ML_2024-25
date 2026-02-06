import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import KFold
import numpy as np


def show_predict_and_real(X, y):
    fig, axes = plt.subplots(nrows=1, ncols=len(y), figsize=(15, 5))
    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        ax.imshow(image.reshape((28, 28)), cmap=plt.cm.gray_r)
        ax.set_title("Predicted: %s" % str(label))


def confusion_matrix(predicted, y):
    confusion_matr = confusion_matrix(y, predicted)
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matr)
    fig, ax = plt.subplots(figsize=(10, 10))
    confusion_matrix_display.plot(ax=ax)


def cross_val(classifier, X, y, k, shuffle=False):
    kf = KFold(n_splits=k, shuffle=shuffle)
    trained_classifier = []
    accuracy_list = []
    for train_index, test_index in kf.split(X):
        X_train_kfold, X_test_kfold = X[train_index], X[test_index]
        y_train_kfold, y_test_kfold = y[train_index], y[test_index]
        classifier_fold = clone(classifier)
        classifier_fold.fit(X_train_kfold, y_train_kfold)
        prediction = classifier_fold.predict(X_test_kfold)
        accuracy = accuracy_score(y_test_kfold, prediction)
        trained_classifier.append(classifier_fold)
        accuracy_list.append(accuracy)
    return trained_classifier, accuracy_list


def check_classifier(classifier, X_train, y_train, X_val, y_val):
    trained_classifier, accuracy_list = cross_val(classifier, X_train, y_train, 5)
    best_classifier = trained_classifier[np.argmax(accuracy_list)]
    accuracy = 0.0
    for i in range(len(accuracy_list)):
        accuracy += accuracy_score(y_val, trained_classifier[i].predict(X_val))
    print(accuracy_list)
    print("Accuracy %.4f" % (accuracy / len(accuracy_list)))
    return best_classifier


def show_classifier_mistakes(clf, X_train, y_train, X_val, y_val, show_number=True):
    best_classifier = check_classifier(clf, X_train, y_train, X_val, y_val)
    prediction = best_classifier.predict(X_val)
    confusion_matrix(y_val, prediction)
    wrong_predictions = [i for i in np.arange(len(prediction)) if prediction[i] != y_val[i]]
    if show_number:
        show_predict_and_real(X_val[wrong_predictions][:8], prediction[wrong_predictions][:8])