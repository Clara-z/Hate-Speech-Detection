import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Given a model prediction on the test set, plot the confusion matrix showing the difference between prediction and the ground truth
def plot_confusion_matrix(y_pred, y_test, labels=[1, 0]):
    # Assuming y_test are the true labels, and y_test_pred are the predicted labels from your model
    # Calculate the confusion matrix and specify the order of labels to match your desired output
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Transpose the confusion matrix to swap the prediction (rows) and actual (columns)
    cm_transposed = cm.T

    # Create a new ConfusionMatrixDisplay instance using the transposed matrix
    # Note: We must manually specify the display labels to match the transposed order
    display_labels = ['Not Hate', 'Hate']  # Note this is unconventional and may be confusing

    # Create the confusion matrix display
    display = ConfusionMatrixDisplay(confusion_matrix=cm_transposed, display_labels=display_labels)

    # Plot the confusion matrix
    display.plot(cmap='Blues', values_format='g')

    # Since we want the x-axis to be the actual labels, we'll set the xlabel to 'Actual label' and the ylabel to 'Predict label'
    plt.xlabel('Actual label')
    plt.ylabel('Predict label')

    # Show the plot
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