import pandas as pd
import logging
import os
from threading import Lock
import numpy as np
import requests
from utils import add_time_features
import joblib
import gc

class DataHandler:
    def __init__(self, api_url=None, data_file=None):
        self.api_url = api_url
        self.data_file = data_file
        self.data_index = 0
        self.lock = Lock()
        self.trade_log = []
        self.historical_data = []
        self.pending_trade = None
        self.pending_order = False
        self.trade_log_file = 'trade_log.csv'

        self.scaler = joblib.load('scaler-v3-07-17.pkl')

        self.instrument_map = {
            'EURUSD-Mini': {'spread': 0.0003, 'fee': 0.0003},
            'GBPUSD-Mini': {'spread': 0.0004, 'fee': 0.0006},
            'USDJPY-Mini': {'spread': 0.03, 'fee': 0.04},
        }

        self.data = self.load_processed_data() if self.data_file else None

    def load_processed_data(self):
        if self.data_file:
            data = pd.read_csv(self.data_file, index_col='Gmt time', parse_dates=['Gmt time'])
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
                        'day_of_week_cos', '10SMA', '50SMA', '14RSI', 'MACD', 'Signal_Line', '10DMA', 
                        '50DMA', '14DMA_RSI']
            data = data[features]
            return data
        else:
            logging.error("Data file not provided")
            return None

    def get_scaled_data(self):
        if self.data is not None:
            scaled_data = self.scaler.transform(self.data.values)
            return scaled_data
        else:
            logging.error("No data available to scale")
            return None

    def get_current_data(self):
        with self.lock:
            if self.data is not None:
                if self.data_index < len(self.data):
                    current_data = self.data.iloc[self.data_index].to_dict()
                    current_data['Gmt time'] = self.data.index[self.data_index]
                    self.historical_data.append(current_data)
                    self.data_index += 1
                    return current_data
                else:
                    logging.warning("Reached the end of the dataset.")
                    return None
            else:
                logging.error("No data available and API URL not provided")
                return None

    def get_historical_data(self, seq_length):
        with self.lock:
            if len(self.historical_data) < seq_length:
                logging.warning(f"Not enough historical data. Needed: {seq_length}, available: {len(self.historical_data)}")
                return []

            historical_data = self.historical_data[-seq_length:]
            historical_df = pd.DataFrame(historical_data)
            if 'Gmt time' in historical_df.columns:
                try:
                    historical_df['Gmt time'] = pd.to_datetime(historical_df['Gmt time'])
                    historical_df.set_index('Gmt time', inplace=True)
                except Exception as e:
                    logging.error("Error setting 'Gmt time' as index in historical data: %s", e)
            else:
                logging.warning("Gmt time missing in historical data, using default index")

            historical_df = add_time_features(historical_df)
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
                        'day_of_week_cos', '10SMA', '50SMA', '14RSI', 'MACD', 'Signal_Line', '10DMA', 
                        '50DMA', '14DMA_RSI']

            if not all(col in historical_df.columns for col in features):
                logging.error(f"Column '{col}' missing in historical data")
                return pd.DataFrame(columns=features)  # Return empty DataFrame with required columns

            return historical_df.to_dict(orient='records')

    def place_order(self, trade, action_type, instrument, size):
        logging.debug(f"DataHandler: Placing order for {instrument}")
        if self.pending_order:
            logging.warning("There is already a pending order")
            return False

        self.pending_order = True

        if instrument not in self.instrument_map:
            logging.error(f"Instrument {instrument} not found in instrument map")
            self.pending_order = False
            return False

        spread = self.instrument_map[instrument]['spread']
        fee = self.instrument_map[instrument]['fee']

        trade['price'] = None  # Price to be set when order is filled
        self.pending_trade = (trade, action_type, instrument, spread, fee, size)
        logging.info(f"Order placed: {trade}, Action: {action_type}, Instrument: {instrument}, Spread: {spread}, Fee: {fee}, Size: {size}")
        logging.debug(f"Pending order set in DataHandler: {self.pending_trade}, {self.pending_order}")
        return True

    def fill_order(self):
        logging.debug("fill_order: Attempting to fill order")
        if not self.pending_order:
            logging.info("fill_order: No pending order to fill")
            return None

        trade, action_type, instrument, spread, fee, size = self.pending_trade
        logging.debug(f"Pending order: {trade}, Action type: {action_type}, Instrument: {instrument}, Spread: {spread}, Fee: {fee}, Size: {size}")
        current_data = self.get_current_data()  # Fetch the most recent data point

        if not current_data:
            logging.error("No current data available for filling order.")
            return None

        logging.debug(f"Current data for filling order: {current_data}")

        if action_type == 'buy':
            trade['price'] = current_data['Open'] + spread
        elif action_type == 'sell':
            trade['price'] = current_data['Open'] - spread
        elif action_type == 'close_long':
            trade['price'] = current_data['Open'] - spread
        elif action_type == 'close_short':
            trade['price'] = current_data['Open'] + spread
        elif action_type == 'reduce_long':
            trade['price'] = current_data['Open'] + spread
        elif action_type == 'reduce_short':
            trade['price'] = current_data['Open'] - spread
        else:
            logging.warning(f"Invalid action type for trade: {action_type}")
            trade['success'] = False
            return trade

        assert trade['price'] is not None, f"Price should not be None for trade: {trade}"

        trade['success'] = True
        logging.debug(f"Trade success: {trade['success']}, Trade: {trade}")

        if trade['success']:
            logging.debug("Order filled successfully")
            self.pending_order = False  # Clear pending order only if filled
            self.pending_trade = None  # Clear pending order only if filled
            self.log_trade(trade)  # Log the trade if it was successful
            logging.info(f"Order filled: {trade}")
        else:
            logging.warning(f"Order failed to fill: {trade}")

        trade['fee'] = fee * size
        trade['size'] = size
        return trade

    def log_trade(self, trade):
        with self.lock:
            self.trade_log.append(trade)
            trade_df = pd.DataFrame([trade])
            trade_df.to_csv(self.trade_log_file, mode='a', header=not os.path.exists(self.trade_log_file), index=False)

    def get_trade_log(self):
        with self.lock:
            return self.trade_log

    def calculate_performance(self):
        return 0

    def reset(self, episode_length):
        logging.info("Resetting DataHandler")
        if self.data is not None:
            self.data_index = np.random.randint(0, len(self.data) - episode_length)
            self.historical_data.clear()
            for i in range(episode_length):
                current_data = self.data.iloc[self.data_index + i].to_dict()
                current_data['Gmt time'] = self.data.index[self.data_index + i]
                self.historical_data.append(current_data)
            logging.info(f"DataHandler reset to new index: {self.data_index}")
        elif self.api_url:
            with self.lock:
                response = requests.post(f"{self.api_url}/reset_data")
                if response.status_code == 200:
                    self.data_index = 0
                    self.historical_data.clear()
                    self.trade_log.clear()
                    self.pending_trade = None
                    self.pending_order = False
                    logging.info("DataHandler reset successfully via API")
                else:
                    logging.error("Failed to reset DataHandler via API")
        else:
            logging.error("No data source provided for reset")
        gc.collect()
