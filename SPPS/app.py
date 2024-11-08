# Install necessary libraries if not already installed


import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os

# Initialize Flask app
app = Flask(__name__)

# Directory to save and load models
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def fetch_stock_data(ticker):
    """Fetch historical stock data for a given ticker."""
    data = yf.download(ticker, start="2015-01-01", end="2023-01-01", interval="1d")
    if 'Close' not in data:
        raise ValueError("Data fetching error. No 'Close' column found.")
    data = data[['Close']]
    return data

def preprocess_data(data):
    """Scale the data and create sequences for LSTM model input."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(ticker):
    """Fetch data, preprocess, train LSTM model, and save the model."""
    data = fetch_stock_data(ticker)
    scaled_data, scaler = preprocess_data(data)
    
    # Split into training and test sets (80-20 split)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    # Prepare sequences
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build and train the model
    model = build_lstm_model((X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=10)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])

    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{ticker}.h5")
    model.save(model_path)
    return model, scaler

def load_or_train_model(ticker):
    """Load a saved model for the ticker or train a new model if not found."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
        data = fetch_stock_data(ticker)
        _, scaler = preprocess_data(data)
    else:
        model, scaler = train_model(ticker)
    return model, scaler

@app.route('/predict/<ticker>', methods=['GET'])
def predict(ticker):
    try:
        # Load or train the model for the given ticker
        model, scaler = load_or_train_model(ticker)

        # Fetch the data and prepare the last 60 days for prediction
        data = fetch_stock_data(ticker)
        scaled_data, _ = preprocess_data(data)
        time_step = 60
        X_test = scaled_data[-time_step:].reshape(1, time_step, 1)  # Last 60 days for prediction

        # Predict the next day
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)  # Scale back to original value
        predicted_price = float(prediction[0][0])  # Convert to native Python float

        return jsonify({"ticker": ticker, "predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
