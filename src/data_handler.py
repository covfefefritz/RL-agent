import pandas as pd
import logging
from threading import Lock
import requests
import time
from utils import add_time_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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

    def place_order(self, trade, action_type, spread):
        logging.debug(f"DataHandler: Placing order")
        if self.pending_order == True:
            logging.warning("There is already a pending order")
            return False

        self.pending_order = True 

        trade['price'] = None  # Price to be set when order is filled
        self.pending_trade = (trade, action_type, spread)
        logging.info(f"Order placed: {trade}, Action: {action_type}, Spread: {spread}")
        logging.debug(f"Pending order set in DataHandler: {self.pending_trade}, {self.pending_order}")
        return True

    # In DataHandler
    def fill_order(self):
        logging.debug("fill_order: Attempting to fill order")
        if self.pending_order == False:
            logging.info("fill_order: No pending order to fill")
            return None

        trade, action_type, spread = self.pending_trade
        logging.debug(f"Pending order: {trade}, Action type: {action_type}, Spread: {spread}")
        current_data = self.get_current_data()  # Fetch the most recent data point

        if not current_data:
            logging.error("No current data available for filling order.")
            return None

        logging.debug(f"Current data for filling order: {current_data}")

        if action_type == 'buy':
            trade['price'] = current_data['Open'] + spread
            success = True
        elif action_type == 'sell':
            trade['price'] = current_data['Open'] - spread
            success = True
        elif action_type == 'close_long':
            trade['price'] = current_data['Open'] - spread
            success = True
        elif action_type == 'close_short':
            trade['price'] = current_data['Open'] + spread
            success = True
        else:
            success = False

        trade['success'] = success
        logging.debug(f"Trade success: {success}, Trade: {trade}")

        if success:
            logging.debug("Order filled successfully")
            self.pending_order = False  # Clear pending order only if filled
            self.pending_trade = None  # Clear pending order only if filled
            self.log_trade(trade)  # Log the trade if it was successful
            logging.info(f"Order filled: {trade}")
        else:
            logging.warning(f"Order failed to fill: {trade}")

        return trade





    def log_trade(self, trade):
        with self.lock:
            self.trade_log.append(trade)
            logging.info(f"Trade logged: {trade}")

    def get_trade_log(self):
        with self.lock:
            return self.trade_log

    def calculate_performance(self):
        return 0

class DataFetcher:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_new_data(self):
        max_retries = 5
        for i in range(max_retries):
            try:
                response = requests.get(self.api_url)
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
