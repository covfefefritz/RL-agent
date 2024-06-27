import pandas as pd
import json
from threading import Lock
import requests
import logging
import time

class DataHandler:
    def __init__(self, api_url):
        self.data_fetcher = DataFetcher(api_url)
        self.index = 0
        self.lock = Lock()
        self.trade_log = []
        self.historical_data = []

    def get_current_data(self):
        with self.lock:
            new_data = self.data_fetcher.fetch_new_data()
            if new_data:
                self.historical_data.append(new_data)
                return new_data
            else:
                return None

    def get_historical_data(self, seq_length):
        with self.lock:
            if len(self.historical_data) < seq_length:
                return self.historical_data
            else:
                return self.historical_data[-seq_length:]

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
                response = requests.get(self.api_url + '/get_data')
                if response.status_code == 404:
                    logging.info("No more data available from API. Exiting.")
                    return None
                response.raise_for_status()
                new_data = response.json()
                if new_data:
                    return new_data
            except requests.RequestException as e:
                logging.warning(f"Request error: {e}. Retrying ({i+1}/{max_retries})...")
                time.sleep(2 ** i)  # Exponential backoff
        logging.error("Max retries exceeded. Exiting.")
        return None
