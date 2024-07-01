import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from utils import add_time_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TradingEnv(gym.Env):
    def __init__(self, data_handler, lstm_predictor, rl_trader, seq_length=60, max_steps=1000):
        super(TradingEnv, self).__init__()
        self.data_handler = data_handler
        self.lstm_predictor = lstm_predictor
        self.rl_trader = rl_trader
        self.seq_length = seq_length
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.done = False
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0,
        }
        logger.info("TradingEnv initialized with seq_length: %d and max_steps: %d", self.seq_length, self.max_steps)

    def reset(self, **kwargs):
        logger.info("Environment reset")
        self.data_handler.index = 0
        historical_data = self.data_handler.get_historical_data(self.seq_length)

        if len(historical_data) < self.seq_length:
            logger.error("Not enough historical data to initialize the environment.")
            return np.zeros(self.observation_space.shape), {}

        self.lstm_predictor.reset()
        for data in historical_data:
            self.lstm_predictor.add_data(data)
        self.lstm_predictor.update_data_and_predict()
        observation = self._get_observation(historical_data[-1])
        logger.debug("Initial observation: %s", observation)
        return observation, {}

    def step(self, action):
        logger.info("Step action received: %d", action)
        current_data = self.data_handler.get_current_data()
        if not current_data:
            logger.warning("No current data available")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        self.lstm_predictor.add_data(current_data)
        self.lstm_predictor.update_data_and_predict()

        state = self._get_observation(current_data)
        action_info = self.rl_trader.perform_action(current_data, action, self.data_handler)
        reward = self.calculate_reward(current_data, action_info)
        self.done = self._check_done()

        # Log performance metrics
        logger.info("Performance metrics:\nTotal trades: %d\nSuccessful trades: %d\nFailed trades: %d\nTotal Profit: %f",
                    self.performance_metrics["total_trades"], 
                    self.performance_metrics["successful_trades"], 
                    self.performance_metrics["failed_trades"],
                    self.performance_metrics["total_profit"])

        logger.debug("New state: %s, Reward: %f, Done: %s", state, reward, self.done)
        return state, reward, self.done, False, {}

    def _get_observation(self, current_data=None):
        if current_data is None:
            logger.warning("Current data is None, returning default state")
            return np.zeros(self.observation_space.shape)

        historical_data = self.data_handler.get_historical_data(self.seq_length)
        if len(historical_data) < self.seq_length:
            logger.error(f"Not enough historical data. Needed: {self.seq_length}, available: {len(historical_data)}")
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

        historical_df = add_time_features(historical_df)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']

        if not all(col in historical_df.columns for col in required_columns):
            logger.error(f"One or more required columns are missing in the historical data: {required_columns}")
            return np.zeros(self.observation_space.shape)

        features = current_data
        prediction = self.lstm_predictor.predictions[-1] if len(self.lstm_predictor.predictions) > 0 else 0.0
        state = np.array([
            features['Open'], features['High'], features['Low'], features['Close'], features['Volume'],
            features['hour_sin'], features['hour_cos'], features['day_of_week_sin'], features['day_of_week_cos'],
            prediction
        ])

        logger.debug("Observation state: %s", state)
        return state

    def calculate_reward(self, current_data, action_info):
        reward = 0
        current_price = current_data['Close']

        if 'action' in action_info:
            if action_info['success'] and action_info['action'] in ['close_long', 'close_short']:
                if self.rl_trader.entry_price is None:
                    logger.error("Entry price is None when trying to close a position. Action: %s", action_info['action'])
                    return 0  # No reward if entry price is not set

                if action_info['action'] == 'close_long':
                    reward = (current_price - self.rl_trader.entry_price) * 10000  # pips for long position
                elif action_info['action'] == 'close_short':
                    reward = (self.rl_trader.entry_price - current_price) * 10000  # pips for short position

                self.performance_metrics["total_trades"] += 1
                if reward > 0:
                    self.performance_metrics["successful_trades"] += 1
                else:
                    self.performance_metrics["failed_trades"] += 1
                self.performance_metrics["total_profit"] += reward

        logger.debug("Reward calculated: %f", reward)
        return reward

    def _check_done(self):
        if self.performance_metrics["total_profit"] < -3000:
            return True
        if self.data_handler.index >= self.max_steps:
            return True
        return False
