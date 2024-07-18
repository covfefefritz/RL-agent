import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# Verify TensorFlow and Keras installation
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

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

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Load the processed data
data = load_processed_data('./../historic_data/processed_EURUSD_Candlestick_1_Hour_BID_13.04.2010-13.04.2024.csv')

# Define the features to be used for scaling and modeling
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
            '10SMA', '50SMA', '14RSI', 'MACD', 'Signal_Line', '10DMA', '50DMA', '14DMA_RSI']

# Ensure the features exist in the data
for feature in features:
    if feature not in data.columns:
        print(f"Feature '{feature}' not found in data columns")

# Extract the feature values
prices = data[features].values

# Fit and transform the scaler on the entire dataset
scaled_features = scaler.fit_transform(prices)

# Confirm the number of features after scaling
print("Scaled features shape (Training):", scaled_features.shape)  # Should print (num_samples, 17) - Adjusted for correct feature count

# Split the data into training (80%) and testing (20%) sets
train_size = int(len(scaled_features) * 0.8)
train_data = scaled_features[:train_size]
test_data = scaled_features[train_size:]

# Create sequences for LSTM
seq_length = 60  # Optimized sequence length

X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Confirm the shape of the training and testing data
print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Define the LSTM model using an Input layer
input_layer = Input(shape=(seq_length, X_train.shape[2]))
x = LSTM(50, return_sequences=True)(input_layer)
x = LSTM(50, return_sequences=False)(x)
x = Dropout(0.2)(x)  # Add dropout layer with dropout rate 0.2
x = Dense(25)(x)
output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Save the trained model and the scaler
model.save('lstm_model-v3-07-17.h5')
joblib.dump(scaler, 'scaler-v3-07-17.pkl')
