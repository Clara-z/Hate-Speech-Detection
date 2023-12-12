# Import necessary libraries
import numpy as np
import pandas as pd
import util
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the data
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
dev_data = np.load('data/hate_speech_dev.npy', allow_pickle=True)
test_data = np.load('data/hate_speech_test.npy', allow_pickle=True)

# Extract the relevant columns
texts_train = train_data[:, 1]
labels_train = train_data[:, 2]

texts_dev = dev_data[:, 1]
labels_dev = dev_data[:, 2]

texts_test = test_data[:, 1]
labels_test = test_data[:, 2]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the vectorizer on the training texts
X_train = tfidf_vectorizer.fit_transform(texts_train)
y_train = labels_train

# Transform the development and test texts
X_dev = tfidf_vectorizer.transform(texts_dev)
y_dev = labels_dev

X_test = tfidf_vectorizer.transform(texts_test)
y_test = labels_test

# Convert labels to integers if they are not already
y_train = y_train.astype(int)
y_dev = y_dev.astype(int)
y_test = y_test.astype(int)

# Define the alpha range
alpha_range = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]
alpha_scores = []

# Evaluate each alpha on the development set
for alpha in alpha_range:
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train, y_train)  # Train on the training set
    y_dev_pred = nb.predict(X_dev)  # Predict on the development set
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)  # Calculate accuracy on the development set
    alpha_scores.append(dev_accuracy)
    print(f"Alpha: {alpha}, Dev Accuracy: {dev_accuracy:.4f}")

# Find the best alpha and its corresponding accuracy
best_index = alpha_scores.index(max(alpha_scores))
best_alpha = alpha_range[best_index]
best_dev_accuracy = alpha_scores[best_index]

print(f"\nBest alpha value: {best_alpha} with Dev Accuracy: {best_dev_accuracy:.4f}\n")

# Train the model with the best alpha on the training set
nb_classifier_best = MultinomialNB(alpha=best_alpha)
nb_classifier_best.fit(X_train, y_train)

# Predict on the test set
y_test_pred = nb_classifier_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Generate and print classification report for the test set
classification_rep = classification_report(y_test, y_test_pred)
print("Classification Report on Test Set:\n", classification_rep)

# # Plot the confusion matrix for the test set (from scratch)
# cm = confusion_matrix(y_test, y_test_pred)
# display = ConfusionMatrixDisplay(confusion_matrix=cm)
# display.plot(cmap='Blues')
# plt.show()

# # Generate Confusion Matrix
# cm = confusion_matrix(y_test, y_test_pred, labels=[1, 0])  # 0 for 'not hate', 1 for 'hate'
# cm_transposed = cm.T
# display = ConfusionMatrixDisplay(confusion_matrix=cm_transposed, display_labels=['Hate', 'Not Hate'])
# display.plot(cmap='Blues')
# plt.xlabel('Actual label')
# plt.ylabel('Predict label')
# plt.show()

# Call util.eval to plot the confusion matrix (assuming util.eval does this)
util.eval(y_test_pred, y_test)
