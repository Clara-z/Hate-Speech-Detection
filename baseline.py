import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import util

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# train data is from hate_speech_train.npy
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
# learn a policy from the training data, using majority rule
# Input: training data
# Output: a value indicating whether the majority of the tweets in the training data are hate speech or not
def majority_rule(train_data):
    """
    Learn a policy from the training data, using majority rule.
    
    Args:
        train_data: training data
    Returns:
        a value indicating whether the majority of the tweets in the training data are hate speech or not
    """
    # get the labels of the training data
    labels = train_data[:, 2]
    # get the number of hate speech tweets in the training data
    num_hate_speech = np.sum(labels == 'hate_speech')
    # get the number of non hate speech tweets in the training data
    num_non_hate_speech = np.sum(labels == 'non_hate_speech')
    # if the number of hate speech tweets is greater than the number of non hate speech tweets, return 'hate_speech'
    if num_hate_speech > num_non_hate_speech:
        return 'hate_speech'
    # otherwise, return 'non_hate_speech'
    else:
        return 'non_hate_speech'

# Use majority rule to determine the label of each tweet
labels = majority_rule(test_data)

# Result
util.plot_confusion_matrix(labels, test_data[:, 2], normalize=True, title='Confusion matrix')
util.print_metrics(labels, test_data[:, 2])

