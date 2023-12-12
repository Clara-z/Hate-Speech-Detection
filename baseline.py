import numpy as np
import pandas as pd
import util

# train data is from hate_speech_train.npy
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
# dev data is from hate_speech_dev.npy
dev_data = np.load('data/hate_speech_dev.npy', allow_pickle=True)
# test data is from hate_speech_test.npy
test_data = np.load('data/hate_speech_test.npy', allow_pickle=True)

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
    num_hate_speech = np.sum(labels == 1)
    # get the number of non hate speech tweets in the training data
    num_non_hate_speech = np.sum(labels == 0)
    # if the number of hate speech tweets is greater than the number of non hate speech tweets, return 'hate_speech'
    if num_hate_speech > num_non_hate_speech:
        return 1
    # otherwise, return 'non_hate_speech'
    else:
        return 0

# Use majority rule to determine the label of each tweet
label = majority_rule(train_data)

# our prediction to the test set is the majority rule
prediction = np.array([label] * len(test_data))

# get the labels of the test set
labels = test_data[:, 2]

# Convert prediction and labels' 1, 0 to 'hate_speech', 'non_hate_speech', respectively
# We do this because the confusion matrix does not accept prediction 1 (nonbinary) and labels 0,1 binary.
prediction_word = ['hate_speech' if p == 1 else 'non_hate_speech' for p in prediction]
labels_word = ['hate_speech' if l == 1 else 'non_hate_speech' for l in labels]

# print the confusion matrix and the metrics
util.plot_confusion_matrix(prediction_word, labels_word)
util.print_metrics(prediction_word, labels_word, average='binary', pos_label='hate_speech')
