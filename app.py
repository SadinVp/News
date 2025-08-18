import pickle
import numpy as np
import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load the saved LSTM model
model = tf.keras.models.load_model(os.path.join("model", "fake_news_lstm_model.keras"))

# Load the tokenizer
with open(os.path.join("model", "tokenizer.pkl"), "rb") as handle:
    tokenizer = pickle.load(handle)

# Max sequence length (must match training)
MAXLEN = 200

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get input text from user
        user_text = request.form["news_text"]

        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

        # Predict with LSTM model
        pred = model.predict(padded)[0][0]

        # Convert to label
        prediction = "🟢 Real News" if pred >= 0.8 else "🔴 Fake News"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
