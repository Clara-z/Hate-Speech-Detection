import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load your dataset
# Assuming dataset is loaded into 'texts' and 'labels'

# Step 1: Data Preprocessing
tokenizer = Tokenizer(num_words=5000)  # Only considering the top 5000 words
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=200)  # Assuming max length of sequences is 200

# Step 2: Building the RNN Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Training the Model
# Split your data into training and testing sets
model.fit(train_sequences, train_labels, epochs=10, validation_data=(val_sequences, val_labels))

# Step 4: Evaluation
test_loss, test_accuracy = model.evaluate(test_sequences, test_labels)
print(f"Test Accuracy: {test_accuracy}")
