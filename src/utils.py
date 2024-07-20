import pandas as pd
import numpy as np
import os
from glob import glob

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

def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_moving_averages_rsi_macd(data, daily_data):
    # Add 10-period and 50-period SMAs
    data['10SMA'] = data['Close'].rolling(window=10).mean()
    data['50SMA'] = data['Close'].rolling(window=50).mean()
    
    # Add 14-period RSI
    data['14RSI'] = calculate_rsi(data, 14)
    
    # Add MACD and Signal Line
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    
    # Add 10-period and 50-period DMAs
    data['10DMA'] = daily_data['Close'].rolling(window=10).mean().reindex(data.index, method='ffill')
    data['50DMA'] = daily_data['Close'].rolling(window=50).mean().reindex(data.index, method='ffill')
    
    # Add 14-day RSI
    daily_rsi = calculate_rsi(daily_data, 14)
    data['14DMA_RSI'] = daily_rsi.reindex(data.index, method='ffill')
    
    return data

def load_and_process_data(file_path, daily_data):
    date_format = "%d.%m.%Y %H:%M:%S.%f"
    data = pd.read_csv(file_path, parse_dates=['Gmt time'], index_col='Gmt time', dayfirst=True)
    data.index = pd.to_datetime(data.index, format=date_format)
    
    data.sort_index(inplace=True)
    
    # Handle missing values
    data.dropna(inplace=True)  # Drop rows with NaN values
    
    # Add cyclical time features
    data = add_time_features(data)
    
    # Add moving averages, RSI, and MACD
    data = add_moving_averages_rsi_macd(data, daily_data)
    
    # Handle missing values created by moving averages, RSI, and MACD
    data.dropna(inplace=True)
    
    return data

def process_and_save_data(file_paths, daily_data):
    for file_path in file_paths:
        # Load and process data
        data = load_and_process_data(file_path, daily_data)
        
        # Get the base filename and create a new filename
        base_filename = os.path.basename(file_path)
        new_filename = f"processed_{base_filename}"
        
        # Save processed data to CSV
        data.to_csv(new_filename)
        print(f"Processed data saved as {new_filename}")

def load_daily_data(instrument):
    daily_file_path = f'./historic_data/daily_data/{instrument}_daily.csv'
    date_format = "%d.%m.%Y"
    daily_data = pd.read_csv(daily_file_path, parse_dates=['Gmt time'], index_col='Gmt time', dayfirst=True)
    daily_data.index = pd.to_datetime(daily_data.index, format=date_format)
    daily_data.sort_index(inplace=True)
    return daily_data

#copy from train_lstm.py
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])  # Assuming 'Close' is the 4th feature (index 3)
    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    # Define the path to unprocessed data
    data_path = './historic_data/unprocessed_data/*.csv'
    
    # Fetch all file paths
    file_paths = glob(data_path)
    
    # Process each instrument separately
    instruments = set([os.path.basename(fp).split('_')[0] for fp in file_paths])
    
    for instrument in instruments:
        # Load daily data for the instrument
        daily_data = load_daily_data(instrument)
        
        # Filter file paths for the current instrument
        instrument_file_paths = [fp for fp in file_paths if os.path.basename(fp).startswith(instrument)]
        
        # Process and save data for the instrument
        process_and_save_data(instrument_file_paths, daily_data)

