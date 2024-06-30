import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from utils import add_time_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class TradingEnv(gym.Env):
    def __init__(self, data_handler, lstm_predictor, seq_length=60):
        super(TradingEnv, self).__init__()
        self.data_handler = data_handler
        self.lstm_predictor = lstm_predictor
        self.seq_length = seq_length
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        logger.info("TradingEnv initialized with seq_length: %d", self.seq_length)

    def reset(self, **kwargs):  # Accept arbitrary keyword arguments
        logger.info("Environment reset")
        self.data_handler.index = 0
        historical_data = self.data_handler.get_historical_data(self.seq_length)
        
        if not historical_data:
            logger.error("Not enough historical data to initialize the environment.")
            return np.zeros(self.observation_space.shape), {}
        
        self.lstm_predictor.data = historical_data
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
        current_price = current_data['Close']
        timestamp = current_data.get('Gmt time', None)
        if timestamp is None:
            logger.error("Timestamp 'Gmt time' is missing from current_data")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        trade = {
            'action': action,
            'price': current_price,
            'timestamp': timestamp,
            'size': 1 if action in [0, 1] else 0
        }
        logger.debug("Trade executed: %s", trade)
        self.data_handler.log_trade(trade)

        done = False  # Define your own condition for termination
        truncated = False  # Define your own condition for truncation
        reward = self.calculate_reward(trade)
        logger.debug("New state: %s, Reward: %f, Done: %s, Truncated: %s", state, reward, done, truncated)
        return state, reward, done, truncated, {}


    def _get_observation(self, current_data=None):
        if current_data is None:
            logger.warning("Current data is None, returning default state")
            return np.zeros(self.observation_space.shape)

        historical_data = self.data_handler.get_historical_data(self.seq_length)
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
        state_features = historical_df[required_columns]

        prediction = self.lstm_predictor.predictions[-1] if len(self.lstm_predictor.predictions) > 0 else 0.0
        state = np.array([
            features['Open'], features['High'], features['Low'], features['Close'], features['Volume'],
            features['hour_sin'], features['hour_cos'], features['day_of_week_sin'], features['day_of_week_cos'], 
            prediction
        ])

        logger.debug("Observation state: %s", state)
        return state
