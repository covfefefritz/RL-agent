import logging
import time
import numpy as np
from stable_baselines3 import PPO
import gc
from rl_env import TradingEnv  # Ensure correct import
from predictor import LSTMPredictor
from data_handler import DataHandler
from rl_trader import LearningTrader

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def train_model(env, model, num_episodes, steps_per_episode):
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Ensure two values are unpacked
        total_reward = 0
        for step in range(steps_per_episode):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
        logger.info(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

        # Explicit garbage collection
        gc.collect()
        logger.info(f"Garbage collection after episode {episode + 1}")

    return rewards

if __name__ == "__main__":
    # Ensure to provide the correct data_file for training
    train_data_handler = DataHandler(data_file='historic_data/processed_EURUSD_Candlestick_5_M_BID_13.07.2021-13.07.2024.csv')
    rl_trader = LearningTrader()

    lstm_predictor = LSTMPredictor(model_path='./lstm_model-07-17-v1_2.h5', scaler_path='./scaler-v3-07-17.pkl')

    # Define number of episodes and steps per episode
    num_episodes = 30
    steps_per_episode = 1500  # Adjusted to cover more data points per episode

    # Training environment
    train_env = TradingEnv(train_data_handler, lstm_predictor, rl_trader, steps_per_episode)

    # Load the existing PPO model
    model_path = "ppo_trading_model-2024-07-15"
    try:
        model = PPO.load(model_path, env=train_env)
        logger.info(f"Loaded existing model from {model_path}")
    except FileNotFoundError:
        model = PPO('MlpPolicy', train_env, verbose=1)
        logger.info(f"Initialized a new model as {model_path} was not found")

    # Train the model
    train_rewards = train_model(train_env, model, num_episodes, steps_per_episode)

    # Save the trained model
    model.save("ppo_forex_model-01")

    # Continuous trading loop
    while True:
        new_data = train_data_handler.get_current_data()
        if new_data is None:
            break
        state = train_env._get_observation(new_data)
        action, _ = model.predict(state)
        train_env.step(action)
        time.sleep(0.05)
