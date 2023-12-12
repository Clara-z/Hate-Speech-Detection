import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Given a model prediction on the test set (np), plot the confusion matrix showing the difference between prediction and the ground truth
# Input: model prediction on the test set
# Output: confusion matrix
def plot_confusion_matrix(np, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Given a model prediction on the test set (np), plot the confusion matrix showing the difference between prediction and the ground truth.
    
    Args:
        np: model prediction on the test set
        classes: classes
        normalize: normalize
        title: title
        cmap: cmap
    """
    cm = confusion_matrix(np, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
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