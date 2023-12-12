import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


def preprocess_data(data):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data[:, 1])
    sequences = tokenizer.texts_to_sequences(data[:, 1])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    return padded_sequences, data[:, 2].astype(int)


# Define the RNN model
def create_rnn_model():
    model = Sequential()
    model.add(Embedding(5000, 64, input_length=200))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_rnn_model(model, train_data, dev_data):
    train_sequences, train_labels = preprocess_data(train_data)
    dev_sequences, dev_labels = preprocess_data(dev_data)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(train_sequences, train_labels, epochs=10, validation_data=(dev_sequences, dev_labels), callbacks=[early_stopping])
    return model


# Load data
# train data is from hate_speech_train.npy
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
# dev data is from hate_speech_dev.npy
dev_data = np.load('data/hate_speech_dev.npy', allow_pickle=True)
# test data is from hate_speech_test.npy
test_data = np.load('data/hate_speech_test.npy', allow_pickle=True)


rnn_model = create_rnn_model()
rnn_model = train_rnn_model(rnn_model, train_data, dev_data)

# Preprocess test data and evaluate the model
test_sequences, test_labels = preprocess_data(test_data)
predictions = rnn_model.predict(test_sequences)
predictions = (predictions > 0.5).astype(np.int)

# Confusion matrix and classification report
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))
