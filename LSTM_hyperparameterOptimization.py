# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbCallback

# Preprocess data
def preprocess_data(data):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data[:, 1])
    sequences = tokenizer.texts_to_sequences(data[:, 1])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    return padded_sequences, data[:, 2].astype(int)

# Initialize WandB and define sweep configuration
wandb.login()
sweep_config = {
    'method': 'random',  # Can be grid, random, or bayes
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'lstm_units': {
            'values': [32, 64, 128]
        },
        'dropout_rate': {
            'values': [0.1, 0.2, 0.3, 0.4]
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="CSCI467", entity="haofengx")

# Define the training function
def train():
    # Initialize a new WandB run
    wandb.init()

    # Get hyperparameters
    config = wandb.config

    # Load data
    # train data is from hate_speech_train.npy
    train_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_train.npy', allow_pickle=True)
    # dev data is from hate_speech_dev.npy
    dev_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_dev.npy', allow_pickle=True)

    # Preprocess data
    train_sequences, train_labels = preprocess_data(train_data)
    dev_sequences, dev_labels = preprocess_data(dev_data)

    # Define the LSTM model with hyperparameters
    model = Sequential()
    model.add(Embedding(5000, 64, input_length=200))
    model.add(LSTM(config.lstm_units, dropout=config.dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(train_sequences, train_labels, epochs=1, batch_size=config.batch_size, validation_data=(dev_sequences, dev_labels), callbacks=[early_stopping, WandbCallback()])

    wandb.finish()

# Run the sweep
wandb.agent(sweep_id, train)