import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import gc  # Garbage collection module
import tensorflow as tf
from predictor import LSTMPredictor

class TradingEnv(gym.Env):
    def __init__(self, data_handler, lstm_predictor, rl_trader, max_steps=2500, seq_length=60, lstm_model_path='./lstm_model-07-17-v1_2.h5', lstm_scaler_path='./scaler-v3-07-17.pkl', instrument='GBPUSD-Mini'):
        super(TradingEnv, self).__init__()
        self.data_handler = data_handler
        self.lstm_predictor = lstm_predictor
        self.rl_trader = rl_trader
        self.seq_length = seq_length
        self.max_steps = max_steps
        self.lstm_model_path = lstm_model_path  # Needed to re-initialize after deletion and GC
        self.lstm_scaler_path = lstm_scaler_path
        self.instrument = instrument

        # Flatten the hierarchical action space into a single Discrete action space
        self.action_space = spaces.Discrete(9)  # 3 action types (Buy, Sell, Hold) * 3 magnitudes (Small, Medium, Large)

        # Update observation space to include new features and LSTM prediction
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        self.episode_step = 0
        self.done = False
        self.truncated = False
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0,
        }
        logging.info("TradingEnv initialized with seq_length: %d and max_steps: %d", self.seq_length, self.max_steps)

    def reset(self, **kwargs):
        logging.info("Environment reset")
        self.episode_step = 0
        self.data_handler.reset(self.max_steps)
        historical_data = [self.data_handler.get_current_data() for _ in range(self.seq_length)]

        if len(historical_data) < self.seq_length:
            logging.error("Not enough historical data to initialize the environment.")
            self.done = True
            self.truncated = False
            return np.zeros(self.observation_space.shape), {}  # Ensure two values are returned

        # Clear and reset the LSTM model
        self._reset_lstm_predictor()

        # Reset RL trader's state
        self.rl_trader.reset()

        # Add historical data to the LSTM predictor
        for data in historical_data:
            self.lstm_predictor.add_data(data)
        self.lstm_predictor.update_data_and_predict()

        observation = self._get_observation(historical_data[-1])
        logging.debug("Initial observation: %s", observation)

        # Reset performance metrics
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0,
        }

        self.done = False
        self.truncated = False
        self.data_handler.pending_order = False
        return observation, {}  # Ensure two values are returned

    def _reset_lstm_predictor(self):
        if hasattr(self, 'lstm_predictor') and self.lstm_predictor is not None:
            self.lstm_predictor.reset()
            del self.lstm_predictor
            gc.collect()
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Re-initialize the LSTM predictor
        self.lstm_predictor = LSTMPredictor(model_path=self.lstm_model_path, scaler_path=self.lstm_scaler_path)

    def step(self, action):
        logging.info("Step action received: %d", action)

        # Decode the combined action into action_type and action_magnitude
        action_type = action // 3
        action_magnitude = action % 3
        magnitudes = {0: 1, 1: 2, 2: 3}
        scaled_action = magnitudes[action_magnitude] if action_type == 0 else -magnitudes[action_magnitude]

        current_data = self.data_handler.get_current_data()
        if not current_data:
            logging.warning("No current data available")
            self.done = True
            self.truncated = False
            return np.zeros(self.observation_space.shape), 0, self.done, self.truncated, {}

        self.lstm_predictor.add_data(current_data)
        self.lstm_predictor.update_data_and_predict()

        state = self._get_observation(current_data)

        if not self.data_handler.pending_order:
            action_info = self.rl_trader.perform_action(current_data, scaled_action, self.data_handler, self.instrument)
            logging.debug("Trade order placed: %s", action_info)
        else:
            logging.debug("Pending order exists. Skipping perform_action.")
            action_info = None

        filled_trade = self.data_handler.fill_order()
        if filled_trade and filled_trade['success']:
            self.rl_trader.update_position(filled_trade)
            logging.debug("Filled trade: %s", filled_trade)
            reward = self.calculate_reward(current_data, filled_trade)
            self.rl_trader.trades.append(filled_trade)
            self.truncated = False  # Assuming truncation is not used in this context
        else:
            logging.debug("No trade was filled. DataHandler pending_order: %s", self.data_handler.pending_order)
            reward = 0

        self.episode_step += 1
        info = {
            "total_trades": self.performance_metrics["total_trades"],
            "successful_trades": self.performance_metrics["successful_trades"],
            "failed_trades": self.performance_metrics["failed_trades"],
            "total_profit": self.performance_metrics["total_profit"]
        }

        logging.info("Performance metrics:\nTotal trades: %d\nSuccessful trades: %d\nFailed trades: %d\nTotal Profit: %f",
                    self.performance_metrics["total_trades"], 
                    self.performance_metrics["successful_trades"], 
                    self.performance_metrics["failed_trades"],
                    self.performance_metrics["total_profit"])

        logging.debug("New state: %s, Reward: %f, Done: %s, Truncated: %s", state, reward, self.done, self.truncated)
        self.done = self._check_done()
        return state, reward, self.done, self.truncated, info

    def _get_observation(self, current_data=None):
        if current_data is None:
            logging.warning("Current data is None, returning default state")
            return np.zeros(self.observation_space.shape)

        historical_data = self.data_handler.get_historical_data(self.seq_length)
        if len(historical_data) < self.seq_length:
            logging.error(f"Not enough historical data. Needed: {self.seq_length}, available: {len(historical_data)}")
            return np.zeros(self.observation_space.shape)

        historical_df = pd.DataFrame(historical_data)

        if 'Gmt time' in historical_df.columns:
            try:
                historical_df['Gmt time'] = pd.to_datetime(historical_df['Gmt time'])
                historical_df.set_index('Gmt time', inplace=True)
            except Exception as e:
                logging.error("Error setting 'Gmt time' as index in historical data: %s", e)
        else:
            logging.warning("Gmt time missing in historical data, using default index")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', '10SMA', '50SMA', '14RSI', 'MACD', 'Signal_Line', '10DMA', '50DMA', '14DMA_RSI']

        if not all(col in historical_df.columns for col in required_columns):
            logging.error(f"One or more required columns are missing in the historical data: {required_columns}")
            return np.zeros(self.observation_space.shape)

        features = current_data
        prediction = self.lstm_predictor.predictions[-1] if len(self.lstm_predictor.predictions) > 0 else 0.0
        state = np.array([
            features['Open'], features['High'], features['Low'], features['Close'], features['Volume'],
            features['hour_sin'], features['hour_cos'], features['day_of_week_sin'], features['day_of_week_cos'],
            features['10SMA'], features['50SMA'], features['14RSI'], features['MACD'], features['Signal_Line'],
            features['10DMA'], features['50DMA'], features['14DMA_RSI'], prediction
        ])

        logging.debug("Observation state: %s", state)
        return state

    def calculate_reward(self, current_data, fill_info):
        reward = 0
        current_price = current_data['Close']
        if fill_info:
            logging.info("Calculate reward: fill info: %s", fill_info)

            action = fill_info['action']
            size = fill_info['size']
            fill_price = fill_info['price']
            fee = fill_info.get('fee', 0)

            # Ensure no entry has None for price
            valid_entries = [entry for entry in self.rl_trader.entry_prices if entry['price'] is not None]
            if not valid_entries:
                avg_entry_price = current_price  # Default to current price if no valid entries
            else:
                entry_prices_sum = sum([entry['price'] * entry['size'] for entry in valid_entries])
                entry_sizes_sum = sum([entry['size'] for entry in valid_entries])
                avg_entry_price = entry_prices_sum / entry_sizes_sum if entry_sizes_sum > 0 else current_price

            # Debug prints
            logging.debug("Current Price: %f", current_price)
            logging.debug("Average Entry Price: %f", avg_entry_price)
            logging.debug("Action: %s, Size: %d, Fill Price: %f, Fee: %f", action, size, fill_price, fee)

            # Calculate reward based on action type
            if action in ['close_long', 'reduce_long']:
                reward = (fill_price - avg_entry_price) * size * 1000 - fee  # Reduced factor to 1000
            elif action in ['close_short', 'reduce_short']:
                reward = (avg_entry_price - fill_price) * size * 1000 - fee  # Reduced factor to 1000
            elif action == 'buy':
                reward = (current_price - avg_entry_price) * size * 1000 - fee  # Reduced factor to 1000
            elif action == 'sell':
                reward = (avg_entry_price - current_price) * size * 1000 - fee  # Reduced factor to 1000

            self.performance_metrics["total_trades"] += 1
            if reward > 0:
                self.performance_metrics["successful_trades"] += 1
            else:
                self.performance_metrics["failed_trades"] += 1
            self.performance_metrics["total_profit"] += reward

        logging.info("Reward calculated: %f", reward)
        return reward

    def _check_done(self):
        if self.performance_metrics["total_profit"] < -1000:
            return True
        logging.debug("Environment step number: %d", self.episode_step)
        if self.episode_step >= self.max_steps:
            return True
        return False
