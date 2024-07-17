import pandas as pd
import logging
import os
from threading import Lock
import requests
import time
from utils import add_time_features

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

class DataHandler:
    def __init__(self, api_url):
        self.api_url = api_url
        self.data_fetcher = DataFetcher(api_url)
        self.index = 0
        self.lock = Lock()
        self.trade_log = []
        self.historical_data = []
        self.pending_trade = None
        self.pending_order = False
        self.trade_log_file = 'trade_log.csv'
        
        # Map to store spreads and fees for different instruments
        self.instrument_map = {
            'EURUSD-Mini': {'spread': 0.0003, 'fee': 0.0003},
            'GBPUSD-Mini': {'spread': 0.0004, 'fee': 0.0006},
            'USDJPY-Mini': {'spread': 0.03, 'fee': 0.04},   
            # Add more instruments as needed
        }

    def get_current_data(self):
        with self.lock:
            new_data = self.data_fetcher.fetch_new_data()
            if new_data:
                df_new_data = pd.DataFrame([new_data])
                try:
                    df_new_data['Gmt time'] = pd.to_datetime(df_new_data['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
                    df_new_data.set_index('Gmt time', inplace=True)
                except Exception as e:
                    logging.error("Error parsing 'Gmt time': %s", e)
                    return None

                df_new_data = add_time_features(df_new_data)
                current_data_dict = df_new_data.iloc[0].to_dict()
                current_data_dict['Gmt time'] = df_new_data.index[0]
                self.historical_data.append(current_data_dict)
                logging.info(f"Fetched current data: {current_data_dict}")
                self.index += 1
                return current_data_dict
            else:
                logging.warning("No new data fetched")
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
            # Ensure the required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']
            for col in required_columns:
                if col not in historical_df.columns:
                    logging.error(f"Column '{col}' missing in historical data")
                    return pd.DataFrame(columns=required_columns)  # Return empty DataFrame with required columns

            return historical_df.to_dict(orient='records')

    def place_order(self, trade, action_type, instrument, size):
        logging.debug(f"DataHandler: Placing order for {instrument}")
        if self.pending_order:
            logging.warning("There is already a pending order")
            return False

        self.pending_order = True 

        # Lookup spread and fee for the instrument
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

        # Ensure that trade has a valid price
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

        # Return trade with size and fee included
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
    
    def reset(self):
        logging.info("Resetting DataHandler")
        with self.lock:
            response = requests.post(f"{self.api_url}/reset_data")
            if response.status_code == 200:
                # response_data = response.json()
                self.index = 0 # response_data['index'] 
                self.historical_data.clear()
                self.trade_log.clear()
                self.pending_trade = None
                self.pending_order = False
                logging.info("DataHandler reset successfully")
            else:
                logging.error("Failed to reset DataHandler")

class DataFetcher:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_new_data(self):
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.api_url}/get_data")
                if response.status_code == 404:
                    logging.info("No more data available from API. Exiting.")
                    return None
                response.raise_for_status()
                new_data = response.json()
                if new_data:
                    # Assume new_data has the required hourly OHLC format
                    logging.debug(f"Fetched new data: {new_data}")
                    time.sleep(0.1)
                    return new_data
            except requests.RequestException as e:
                logging.warning(f"Request error: {e}. Retrying ({i+1}/{max_retries})...")
                time.sleep(2 ** i)  # Exponential backoff
        logging.error("Max retries exceeded. Exiting.")
        return None