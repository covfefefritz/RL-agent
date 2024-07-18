import pandas as pd
import numpy as np

def add_time_features(data):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    # Cyclical encoding for hour and day of week
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    data.drop(columns=['hour', 'day_of_week'], inplace=True)
    return data

def load_data(file_path, scaler):
    date_format = "%d.%m.%Y %H:%M:%S.%f"
    data = pd.read_csv(file_path, parse_dates=['Gmt time'], index_col='Gmt time', dayfirst=True)
    data.index = pd.to_datetime(data.index, format=date_format)
    
    data.sort_index(inplace=True)
    
    # Handle missing values
    data.dropna(inplace=True)  # Drop rows with NaN values
    
    # Add cyclical time features
    data = add_time_features(data)
    
    # Define the features to be used for scaling and modeling
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
    prices = data[features].values
    
    # Fit and transform the scaler on the entire dataset
    scaled_prices = scaler.fit_transform(prices)
    
    return scaled_prices, data

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])  # Assuming 'Close' is the 4th feature (index 3)
    return np.array(sequences), np.array(labels)
