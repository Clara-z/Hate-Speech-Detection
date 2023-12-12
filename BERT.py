import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import util

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

def create_bert_model():
    # Loading the pre-trained BERT model for sequence classification
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    # Initializing the optimizer with a specific learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    # Defining the loss function for a classification problem with logits
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Compiling the model with the optimizer, loss function, and metric to track
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Loading training, development, and test data from numpy files
train_data = np.load('data/hate_speech_train.npy', allow_pickle=True)
dev_data = np.load('data/hate_speech_dev.npy', allow_pickle=True)
test_data = np.load('data/hate_speech_test.npy', allow_pickle=True)

# Initializing the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encoding the datasets using the bert_encode function
train_inputs, train_masks = bert_encode(train_data[:, 1])
train_labels = np.array(train_data[:, 2]).astype(int)

dev_inputs, dev_masks = bert_encode(dev_data[:, 1])
dev_labels = np.array(dev_data[:, 2]).astype(int)

test_inputs, test_masks = bert_encode(test_data[:, 1])
test_labels = np.array(test_data[:, 2]).astype(int)

# Printing the shapes of training inputs and masks for verification
print("Train Inputs Shape:", train_inputs.shape)
print("Train Masks Shape:", train_masks.shape)

# Creating the BERT model
bert_model = create_bert_model()

# Setting up early stopping to prevent overfitting during training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the BERT model using the training and validation data
bert_history = bert_model.fit(
    [train_inputs, train_masks], train_labels,
    validation_data=([dev_inputs, dev_masks], dev_labels),
    epochs=3,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluating the model on the test dataset
test_loss, test_acc = bert_model.evaluate([test_inputs, test_masks], test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Making predictions on the test dataset
predictions = bert_model.predict([test_inputs, test_masks])[0]
# Converting the logits to class indexes
predictions = np.argmax(predictions, axis=1)

# Using a custom evaluation function from the 'util' module to evaluate the predictions
util.eval(predictions, test_labels)