import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def build_and_train_model(texts, labels):
    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=200)

    # Build the LSTM model
    model = Sequential([
        Embedding(10000, 128),
        LSTM(64),
        Dense(1, activation='sigmoid') # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, np.array(labels), epochs=5, validation_split=0.2)
    
    return model, tokenizer

# Example usage (in a Jupyter Notebook for instance)
# model, tokenizer = build_and_train_model(texts, labels)
# model.save('models/sentiment_model.h5')
