import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def confusion_matrix_class(matrice_confusione, classe_di_interesse):
    # Trova l'indice della classe di interesse nella matrice di confusione
    i = classe_di_interesse - 1

    tp = matrice_confusione[i, i]
    tn = np.sum(np.diag(matrice_confusione)) - tp
    fp = np.sum(matrice_confusione[:, i]) - tp
    fn = np.sum(matrice_confusione[i, :]) - tp

    accuracy = (tp + tn)/(tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # matrix_confusion = np.array([[tp, fp], [fn, tn]])

    return accuracy, precision, recall


def calcolateModel(y_test, y_pred, classes, flagplt, strName):
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = np.matrix(cm)
    perfMatrix = np.zeros((4, 3))
    nClasses = len(classes)
    for i in range(1, nClasses + 1):
        accuracy, precision, recall = confusion_matrix_class(cm, i)
        perfMatrix[i-1, :] = [accuracy, precision, recall]

    # Visualize the confusion matrix using Matplotlib
    if flagplt:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        # Add labels and title to the plot
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix' + strName)
        # Show the plot
        plt.show()

    return perfMatrix
