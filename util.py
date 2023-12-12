import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Given a model prediction on the test set, plot the confusion matrix showing the difference between prediction and the ground truth
def plot_confusion_matrix(y_pred, y_test):
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_transposed = cm.T
    display = ConfusionMatrixDisplay(confusion_matrix=cm_transposed, display_labels=['Hate', 'Not Hate'])
    display.plot(cmap='Blues')
    plt.xlabel('Actual label')
    plt.ylabel('Predict label')

    plt.show()


# Return the accuracy, precision, recall, and F1 score of the model prediction on the test set
# Input: model prediction on the test set, and the ground truth
# Output: accuracy, precision, recall, and F1 score
def get_metrics(y_pred, y_test, average='binary', pos_label=1):
    """
    Return the accuracy, precision, recall, and F1 score of the model prediction on the test set.
    
    Args:
        y_pred: model prediction on the test set
        y_test: the ground truth
    Returns:
        accuracy, precision, recall, and F1 score
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, pos_label=pos_label)
    recall = recall_score(y_test, y_pred, average=average, pos_label=pos_label)
    f1 = f1_score(y_test, y_pred, average=average, pos_label=pos_label)
    return accuracy, precision, recall, f1

def print_metrics(y_pred, y_test, average='binary', pos_label=1):
    """
    Print the accuracy, precision, recall, and F1 score of the model prediction on the test set.
    
    Args:
        accuracy: accuracy
        precision: precision
        recall: recall
        f1: F1 score
    """
    accuracy, precision, recall, f1 = get_metrics(y_pred, y_test, average=average, pos_label=pos_label)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)

def eval(y_pred, y_test):
    """
    Print the confusion matrix and the metrics of the model prediction on the test set.
    
    Args:
        y_pred: model prediction on the test set
        y_test: the ground truth
    """
    plot_confusion_matrix(y_pred, y_test)
    print_metrics(y_pred, y_test)