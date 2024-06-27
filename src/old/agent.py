import os
import requests
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from datetime import datetime
from utils import create_sequences, add_time_features
import time
import tensorflow as tf
import logging

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingAgent:
    def __init__(self, model_path, scaler_path, seq_length=60, max_positions=3, spread=0.0002, api_url='', 
                 stop_loss=0.95, buy_threshold=1.0025, sell_threshold=0.9975, prediction_steps=15):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.seq_length = seq_length
        self.max_positions = max_positions
        self.spread = spread
        self.trading_log = []
        self.data = []
        self.predictions = np.array([])  # Initialize as an empty numpy array
        self.api_url = api_url
        self.long_positions = 0
        self.short_positions = 0
        self.long_trades = []  # Store long trades with size and price
        self.short_trades = []  # Store short trades with size and price

        # Parameters for trading
        self.stop_loss = stop_loss
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prediction_steps = prediction_steps

        # For cumulative profit/loss calculation
        self.cumulative_profit_loss = 0
        self.max_drawdown = 0
        self.peak_profit_loss = 0
        self.number_of_trades = 0  # Initialize trade counter

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
                if new_data:  # Ensure new_data is not empty
                    self.data.append(new_data)
                    return new_data
            except requests.RequestException as e:
                logging.warning(f"Request error: {e}. Retrying ({i+1}/{max_retries})...")
                time.sleep(2 ** i)  # Exponential backoff
        logging.error("Max retries exceeded. Exiting.")
        return None
    
    def update_max_drawdown(self):
        if self.cumulative_profit_loss > self.peak_profit_loss:
            self.peak_profit_loss = self.cumulative_profit_loss
        drawdown = self.peak_profit_loss - self.cumulative_profit_loss
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def preprocess_data(self):
        if len(self.data) < self.seq_length + 1:
            return None

        df = pd.DataFrame(self.data)
        try:
            df['Gmt time'] = pd.to_datetime(df['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")  # Ensure 'Gmt time' is parsed correctly
        except ValueError as e:
            logging.error(f"Error parsing datetime: {e}")
            return None
        df.index = df['Gmt time']  # Set 'Gmt time' as the index

        # Add cyclical time features
        df = add_time_features(df)

        # Ensure only the necessary columns are used
        features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']]
        return features

    def update_data_and_predict(self):
        features = self.preprocess_data()
        if features is None or features.shape[0] < self.seq_length:
            logging.warning(f"Not enough data to create sequences. Needed: {self.seq_length}, available: {features.shape[0] if features is not None else 0}")
            return

        df_scaled = self.scaler.transform(features.values)
        X_new, _ = create_sequences(df_scaled, self.seq_length)

        if len(X_new) == 0:
            logging.error("No sequences created. Check the data and sequence creation logic.")
            return

        # Debug: Print input sequence
        logging.debug(f"Input sequence shape: {X_new[-1].shape}")
        logging.debug(f"Input sequence: {X_new[-1]}")

        # Perform recursive prediction
        last_sequence = X_new[-1]
        recursive_predictions = self.recursive_predict(last_sequence, steps=self.prediction_steps)

        # Use the prediction furthest into the future
        self.predictions = recursive_predictions[-1]

        # Debug: Print predictions
        logging.debug(f"Recursive prediction: {recursive_predictions}")

    def recursive_predict(self, sequence, steps=1):
        sequence = sequence.copy()
        predictions = []

        for step in range(steps):
            # Debug: Print shape before prediction
            logging.debug(f"Step {step+1}/{steps}: Sequence shape before prediction: {sequence[np.newaxis, :, :].shape}")

            prediction = self.model.predict(sequence[np.newaxis, :, :])[0, 0]
            predictions.append(prediction)

            # Prepare the next sequence
            sequence = np.roll(sequence, -1, axis=0)
            sequence[-1, 3] = prediction  # Assuming 'Close' is the 4th feature (index 3)

            # Debug: Print updated sequence
            logging.debug(f"Step {step+1}/{steps}: Updated sequence: {sequence[-1]}")

        # Inverse transform the predictions
        prediction_scaled = np.zeros((len(predictions), sequence.shape[1]))
        prediction_scaled[:, 3] = predictions  # Assuming 'Close' price is the last feature
        inverse_transformed = self.scaler.inverse_transform(prediction_scaled)

        return inverse_transformed[:, 3]  # Return the 'Close' price predictions

    def log_action(self, action, price, timestamp, size=1, entry_price=None, exit_price=None, weighted_performance=None):
        self.cumulative_profit_loss += weighted_performance or 0
        self.update_max_drawdown()
        if action in ['pred. entry buy', 'pred. entry sell']:
            self.number_of_trades += 1

        log_entry = {
            'action': action,
            'price': price,
            'timestamp': timestamp,
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'weighted_performance': weighted_performance,
            'net_position': self.long_positions - self.short_positions,
            'cumulative_profit_loss': self.cumulative_profit_loss,
            'max_drawdown': self.max_drawdown,
            'number_of_trades': self.number_of_trades
        }
        self.trading_log.append(log_entry)
        logging.info(f"Logged action: {log_entry}")
        self.save_trading_log()

    def save_trading_log(self, file_path='trading_log.csv'):
        pd.DataFrame(self.trading_log).to_csv(file_path, index=False)

    def calculate_average_entry_price(self, trades):
        if not trades:
            return 0
        entry_prices = [trade['price'] for trade in trades]
        return sum(entry_prices) / len(entry_prices)

    def calculate_position_performance(self, trades, current_price):
        if not trades:
            return 0
        average_entry_price = self.calculate_average_entry_price(trades)
        position_percentage = (current_price - average_entry_price) / average_entry_price
        total_size = sum(trade['size'] for trade in trades)
        weighted_performance = position_percentage * total_size
        return weighted_performance

    def aggregate_trades(self, trades, current_price, action):
        if not trades:
            return 0, 0, 0
        entry_prices = [trade['price'] for trade in trades]
        average_entry_price = sum(entry_prices) / len(entry_prices)
        total_size = sum(trade['size'] for trade in trades)
        weighted_performance = self.calculate_position_performance(trades, current_price)
        return average_entry_price, total_size, weighted_performance

    def check_positions(self):
        if not self.data:
            logging.warning("No data available to check positions.")
            return
        current_price = self.data[-1]['Close']  # Use actual current price without spread
        timestamp = self.data[-1]['Gmt time']

        # Check long positions
        if self.long_trades:
            avg_long_entry_price = self.calculate_average_entry_price(self.long_trades)
            if current_price < avg_long_entry_price * self.stop_loss:
                avg_entry, total_size, weighted_performance = self.aggregate_trades(self.long_trades, current_price, 'stop loss sell')
                self.log_action('stop loss sell', current_price, timestamp, total_size, avg_entry, current_price, weighted_performance)
                logging.info(f"Stop loss triggered: Sold at {current_price}")
                self.long_positions -= total_size
                self.long_trades.clear()  # Clear all long trades

        # Check short positions
        if self.short_trades:
            avg_short_entry_price = self.calculate_average_entry_price(self.short_trades)
            if current_price > avg_short_entry_price / self.stop_loss:
                avg_entry, total_size, weighted_performance = self.aggregate_trades(self.short_trades, current_price, 'stop loss buy')
                self.log_action('stop loss buy', current_price, timestamp, total_size, avg_entry, current_price, weighted_performance)
                logging.info(f"Stop loss triggered: Bought at {current_price}")
                self.short_positions -= total_size
                self.short_trades.clear()  # Clear all short trades

    def execute_trades(self):
        if self.predictions.size == 0:
            logging.warning("No predictions available.")
            return

        pred = self.predictions  # Use the furthest prediction
        current_price = self.data[-1]['Close']  # Use actual current price without spread
        timestamp = self.data[-1]['Gmt time']  # Log the actual time for backtesting purposes

        logging.info(f"Prediction for {timestamp}: {pred}")
        logging.info(f"Current market price: {current_price}")

        # Close positions if the prediction indicates an opposite move
        if self.long_trades:
            avg_long_entry_price = self.calculate_average_entry_price(self.long_trades)
            if pred < current_price * self.sell_threshold:
                avg_entry, total_size, weighted_performance = self.aggregate_trades(self.long_trades, current_price, 'opposite move sell')
                self.log_action('opposite move sell', current_price, timestamp, total_size, avg_entry, current_price, weighted_performance)
                logging.info(f"Opposite move predicted: Sell logged at {current_price}")
                self.long_positions -= total_size
                self.long_trades.clear()  # Clear all long trades

        if self.short_trades:
            avg_short_entry_price = self.calculate_average_entry_price(self.short_trades)
            if pred > current_price * self.buy_threshold:
                avg_entry, total_size, weighted_performance = self.aggregate_trades(self.short_trades, current_price, 'opposite move buy')
                self.log_action('opposite move buy', current_price, timestamp, total_size, avg_entry, current_price, weighted_performance)
                logging.info(f"Opposite move predicted: Buy logged at {current_price}")
                self.short_positions -= total_size
                self.short_trades.clear()  # Clear all short trades

        # Open new positions if allowed
        if self.long_positions < self.max_positions and self.short_positions == 0:
            if pred > current_price * self.buy_threshold:
                buy_price = current_price * (1 + self.spread)  # Apply spread for new buy
                size = 1  # Adjust size as needed
                self.log_action('pred. entry buy', buy_price, timestamp, size)
                logging.info(f"Buy logged at {buy_price}")
                self.long_positions += size
                self.long_trades.append({'price': buy_price, 'size': size})

        if self.short_positions < self.max_positions and self.long_positions == 0:
            if pred < current_price * self.sell_threshold:
                sell_price = current_price * (1 - self.spread)  # Apply spread for new sell
                size = 1  # Adjust size as needed
                self.log_action('pred. entry sell', sell_price, timestamp, size)
                logging.info(f"Sell logged at {sell_price}")
                self.short_positions += size
                self.short_trades.append({'price': sell_price, 'size': size})

        # Check existing positions for stop loss
        self.check_positions()

if __name__ == "__main__":
    agent = TradingAgent(
        model_path='lstm_model_v3_simple.h5',
        scaler_path='scaler_v3_simple.pkl',
        api_url='http://api:5000/get_data',
        stop_loss=0.95,
        buy_threshold=1.0025,
        sell_threshold=0.9975,
        prediction_steps=15
    )
    
    try:
        while True:
            new_data = agent.fetch_new_data()
            if new_data is None:
                break  # Exit the loop if no more data is available
            agent.update_data_and_predict()
            agent.execute_trades()
            time.sleep(0.05)  # Wait for the next data point
    except KeyboardInterrupt:
        logging.info("Terminating the agent.")
