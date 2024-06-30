import pandas as pd
import logging
from threading import Lock
import requests
import time
from utils import add_time_features

class DataHandler:
    def __init__(self, api_url):
        self.api_url = api_url
        self.data_fetcher = DataFetcher(api_url)
        self.index = 0
        self.lock = Lock()
        self.trade_log = []
        self.historical_data = []

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
                return current_data_dict
            else:
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

    def log_trade(self, trade):
        with self.lock:
            self.trade_log.append(trade)

    def get_trade_log(self):
        with self.lock:
            return self.trade_log

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
                    logging.debug(f"Fetched new data: {new_data}")
                    time.sleep(0.1)
                    return new_data
            except requests.RequestException as e:
                logging.warning(f"Request error: {e}. Retrying ({i+1}/{max_retries})...")
                time.sleep(2 ** i)  # Exponential backoff
        logging.error("Max retries exceeded. Exiting.")
        return None
