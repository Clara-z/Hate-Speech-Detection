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
lstm_units_options = [32, 64, 128]
dropout_rate_options = [0.1, 0.2, 0.3, 0.4]


def preprocess_data(data):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data[:, 1])
    sequences = tokenizer.texts_to_sequences(data[:, 1])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    return padded_sequences, data[:, 2].astype(int)


# Define the LSTM model without hyperparameter tuning
# def create_lstm_model():
#     model = Sequential()
#     model.add(Embedding(5000, 64, input_length=200))
#     model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# Define the LSTM model with hyperparameter tuning
def create_lstm_model(lstm_units, dropout_rate):
    model = Sequential()
    model.add(Embedding(5000, 64, input_length=200))
    model.add(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# def train_lstm_model(model, train_data, dev_data):
#     train_sequences, train_labels = preprocess_data(train_data)
#     dev_sequences, dev_labels = preprocess_data(dev_data)

#     early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#     model.fit(train_sequences, train_labels, epochs=10, validation_data=(dev_sequences, dev_labels), callbacks=[early_stopping])
#     return model



# Load data
# train data is from hate_speech_train.npy
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
# dev data is from hate_speech_dev.npy
dev_data = np.load('data/hate_speech_dev.npy', allow_pickle=True)
# test data is from hate_speech_test.npy
test_data = np.load('data/hate_speech_test.npy', allow_pickle=True)


# rnn_model = create_lstm_model()
# rnn_model = train_lstm_model(rnn_model, train_data, dev_data)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

best_val_accuracy = 0
best_lstm_units = 0
best_dropout_rate = 0

train_sequences, train_labels = preprocess_data(train_data)
dev_sequences, dev_labels = preprocess_data(dev_data)

for lstm_units in lstm_units_options:
    for dropout_rate in dropout_rate_options:
        print(f"Training with {lstm_units} LSTM units and {dropout_rate} dropout rate")

        # Create and train the model
        lstm_model = create_lstm_model(lstm_units, dropout_rate)
        lstm_model.fit(train_sequences, train_labels, epochs=10, validation_data=(dev_sequences, dev_labels), steps_per_epoch=300, callbacks=[early_stopping], verbose=1)

        # Evaluate the model
        val_accuracy = lstm_model.evaluate(dev_sequences, dev_labels, verbose=1)[1]

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_lstm_units = lstm_units
            best_dropout_rate = dropout_rate

        print(f"Validation accuracy: {val_accuracy}")

print(f"Best LSTM units: {best_lstm_units}, Best Dropout rate: {best_dropout_rate}, Best Validation Accuracy: {best_val_accuracy}")


# Preprocess test data and evaluate the model
test_sequences, test_labels = preprocess_data(test_data)
predictions = lstm_model.predict(test_sequences)
predictions = (predictions > 0.5).astype(int)

util.eval(predictions, test_labels)
