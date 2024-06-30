import pandas as pd
import numpy as np

def add_time_features(df):
    # Ensure the index is a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df.index.dayofweek

    # Cyclical encoding for hour and day of week
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df.drop(columns=['hour', 'day_of_week'], inplace=True)
    return df

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])  # Assuming 'Close' is the 4th feature (index 3)
    return np.array(sequences), np.array(labels)

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
