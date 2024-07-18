import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# Verify TensorFlow and Keras installation
import tensorflow as tf
from tensorflow import keras

#print("TensorFlow version:", tf.__version__)
#print("Keras version:", keras.__version__)

# Define function to load processed data
def load_processed_data(file_path):
    data = pd.read_csv(file_path, index_col='Gmt time', parse_dates=['Gmt time'])
    return data

# Define function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])  # Assuming 'Close' is the 4th feature (index 3)
    return np.array(sequences), np.array(labels)

# Load the pre-trained model and scaler
model = load_model('lstm_model-v3-07-17-v2.h5')
scaler = joblib.load('scaler-v3-07-17.pkl')

# Load the new processed data
data = load_processed_data('historic_data/processed_USDJPY_Candlestick_5_M_BID_13.07.2021-31.10.2022.csv')

# Define the features to be used for scaling and modeling
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            '10SMA', '50SMA', '14RSI', 'MACD', 'Signal_Line', '10DMA', '50DMA', '14DMA_RSI']

# Ensure the features exist in the data
for feature in features:
    if feature not in data.columns:
        print(f"Feature '{feature}' not found in data columns")

# Extract the feature values
prices = data[features].values

# Scale the new data using the pre-trained scaler
scaled_features = scaler.transform(prices)

# Confirm the number of features after scaling
print("Scaled features shape (New Data):", scaled_features.shape)  # Should print (num_samples, 17)

# Split the new data into training (80%) and testing (20%) sets
train_size = int(len(scaled_features) * 0.8)
train_data = scaled_features[:train_size]
test_data = scaled_features[train_size:]

# Create sequences for LSTM
seq_length = 60  # Optimized sequence length

X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Confirm the shape of the training and testing data
print(f"Training data shape (New Data): X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing data shape (New Data): X_test: {X_test.shape}, y_test: {y_test.shape}")

# Continue training the pre-trained model with the new data
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Save the updated model and the scaler
model.save('lstm_model-07-17-v1_2.h5')
joblib.dump(scaler, 'scaler-v3-07-17.pkl')

