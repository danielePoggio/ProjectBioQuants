import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def calcolateModel(y_test, y_pred, classes):
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the precision of the model
    precision = precision_score(y_test, y_pred, average='micro')
    # Calculate the recall of the model
    recall = recall_score(y_test, y_pred, average='micro')
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Visualize the confusion matrix using Matplotlib
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # Add labels and title to the plot
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    # Show the plot
    plt.show()
    return accuracy, precision, recall, cm
