import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Given a model prediction on the test set, plot the confusion matrix showing the difference between prediction and the ground truth
def plot_confusion_matrix(y_pred, y_test):
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1,0])
    cm_transposed = cm.T
    display = ConfusionMatrixDisplay(confusion_matrix=cm_transposed, display_labels=['Hate', 'Not Hate'])
    display.plot(cmap='Blues')
    plt.xlabel('Actual label')
    plt.ylabel('Predict label')

    plt.show()

# Return the accuracy, precision, recall, and F1 score of the model prediction on the test set
# Input: model prediction on the test set
# Output: accuracy, precision, recall, and F1 score
def get_metrics(np):
    """
    Return the accuracy, precision, recall, and F1 score of the model prediction on the test set.
    
    Args:
        np: model prediction on the test set
    Returns:
        accuracy, precision, recall, and F1 score
    """
    accuracy = accuracy_score(np, labels)
    precision = precision_score(np, labels, average='macro')
    recall = recall_score(np, labels, average='macro')
    f1 = f1_score(np, labels, average='macro')
    return accuracy, precision, recall, f1

def print_metrics(np):
    """
    Print the accuracy, precision, recall, and F1 score of the model prediction on the test set.
    
    Args:
        accuracy: accuracy
        precision: precision
        recall: recall
        f1: F1 score
    """
    accuracy, precision, recall, f1 = get_metrics(np)
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)