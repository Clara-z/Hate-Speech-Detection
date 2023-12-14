import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbCallback

wandb.login()
sweep_config = {
    'method': 'random',  # You can use grid, random, or bayesian optimization
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-5,
            'max': 5e-5
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256]
        },
        'optimizer_name':{
            'values': ['adam', 'sgd', 'rmsprop']
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="CSCI467", entity="haofengx")

# Initializing the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

def bert_encode(data, max_length=200):
    input_ids = []  # Initializing a list to store encoded input IDs
    attention_masks = []  # Initializing a list to store attention masks

    # Looping over each item in the data
    for i in range(len(data)):
        # Encoding the text data to BERT's format including special tokens, padding, and attention masks
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,  # Adding special tokens (like [CLS] and [SEP])
            max_length=max_length,  # Setting maximum sequence length
            pad_to_max_length=True,  # Enabling padding to max_length
            return_attention_mask=True,  # Including attention masks in the output
            return_tensors='np',  # Returning numpy tensors
        )
        
        # Appending the encoded input ID and attention mask to their respective lists
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])

    # Converting lists of input IDs and attention masks to numpy arrays and returning them
    return np.array(input_ids), np.array(attention_masks)

def create_bert_model(optimizer):
    # Loading the pre-trained BERT model for sequence classification
    model = TFBertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', from_pt=True)    # Initializing the optimizer with a specific learning rate
    # Defining the loss function for a classification problem with logits
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Compiling the model with the optimizer, loss function, and metric to track
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def train():
    # Initialize a new WandB run
    wandb.init()

    # Loading training, development, and test data from numpy files
    train_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_train.npy', allow_pickle=True)
    # dev data is from hate_speech_dev.npy
    dev_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_dev.npy', allow_pickle=True)

    # Encoding the datasets using the bert_encode function
    train_inputs, train_masks = bert_encode(train_data[:, 1])
    train_labels = np.array(train_data[:, 2]).astype(int)

    dev_inputs, dev_masks = bert_encode(dev_data[:, 1])
    dev_labels = np.array(dev_data[:, 2]).astype(int)

    # Get hyperparameters from WandB
    config = wandb.config
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    optimizer_name = config.optimizer_name

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    # Create the BERT model
    bert_model = create_bert_model(optimizer)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    bert_history = bert_model.fit(
        [train_inputs, train_masks], train_labels,
        validation_data=([dev_inputs, dev_masks], dev_labels),
        epochs=1,
        batch_size=batch_size,
        callbacks=[early_stopping, WandbCallback()]
    )

    # Close the WandB run
    wandb.finish()

# Run the sweep
wandb.agent(sweep_id, train, count=60)