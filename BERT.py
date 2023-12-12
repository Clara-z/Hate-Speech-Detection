import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import util

def bert_encode(data, max_length=200):
    input_ids = []
    attention_masks = []

    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='np', 
        )
        
        input_ids.append(encoded['input_ids'][0]) 
        attention_masks.append(encoded['attention_mask'][0])

    return np.array(input_ids), np.array(attention_masks)

def create_bert_model():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Load data
# train data is from hate_speech_train.npy
train_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_train.npy', allow_pickle=True)
# dev data is from hate_speech_dev.npy
dev_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_dev.npy', allow_pickle=True)
# test data is from hate_speech_test.npy
test_data = np.load('/content/drive/MyDrive/CSCI467/data/hate_speech_test.npy', allow_pickle=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_inputs, train_masks = bert_encode(train_data[:, 1])
train_labels = np.array(train_data[:, 2]).astype(int)

dev_inputs, dev_masks = bert_encode(dev_data[:, 1])
dev_labels = np.array(dev_data[:, 2]).astype(int)

test_inputs, test_masks = bert_encode(test_data[:, 1])
test_labels = np.array(test_data[:, 2]).astype(int)

print("Train Inputs Shape:", train_inputs.shape)
print("Train Masks Shape:", train_masks.shape)

bert_model = create_bert_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

bert_history = bert_model.fit(
    [train_inputs, train_masks], train_labels,
    validation_data=([dev_inputs, dev_masks], dev_labels),
    epochs=3,
    batch_size=32,
    callbacks=[early_stopping]
)

test_loss, test_acc = bert_model.evaluate([test_inputs, test_masks], test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

predictions = bert_model.predict([test_inputs, test_masks])[0]
predictions = np.argmax(predictions, axis=1)

util.eval(predictions, test_labels)