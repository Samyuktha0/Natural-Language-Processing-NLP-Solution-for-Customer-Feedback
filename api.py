from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('models/sentiment_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    text_to_analyze = data['text']

    # Preprocess the text
    sequence = tokenizer.texts_to_sequences([text_to_analyze])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    # Make a prediction
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"

    return jsonify({"sentiment": sentiment, "score": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
