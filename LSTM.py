import numpy as np
import pandas as pd
import util
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# Hyperparameter grid


def preprocess_data(data):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data[:, 1])
    sequences = tokenizer.texts_to_sequences(data[:, 1])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    return padded_sequences, data[:, 2].astype(int)


# Load data
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
dev_data = np.load('data/hate_speech_dev.npy', allow_pickle=True)
test_data = np.load('data/hate_speech_test.npy', allow_pickle=True)

# Preprocess data
train_sequences, train_labels = preprocess_data(train_data)
dev_sequences, dev_labels = preprocess_data(dev_data)
test_sequences, test_labels = preprocess_data(test_data)


# Define the LSTM model with best hyperparameters
model = Sequential()
model.add(Embedding(5000, 64, input_length=200))
model.add(LSTM(64, dropout=0.1))  # Best hyperparameters
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit(train_sequences, train_labels, epochs=10, batch_size=32, validation_data=(dev_sequences, dev_labels), callbacks=[early_stopping], verbose=1)

# Evaluate the model
predictions = model.predict(test_sequences)
predictions = (predictions > 0.5).astype(int)

util.eval(predictions, test_labels)

# Identifying misclassified samples
misclassified_indices = np.where(predictions != test_labels.reshape(-1, 1))[0]

# Randomly select 50 misclassified samples
np.random.shuffle(misclassified_indices)
selected_misclassified_indices = misclassified_indices[:50]

# Outputting 50 misclassified samples
print("50 Randomly Selected Misclassified Samples:")
for index in selected_misclassified_indices:
    print(f"Sample Index: {index}, text: {test_data[index][1]}, Predicted Label: {predictions[index][0]}, Actual Label: {test_labels[index]}")