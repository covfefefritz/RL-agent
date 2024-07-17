import logging
import time
import numpy as np
from stable_baselines3 import PPO
from tensorflow.keras.models import load_model
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
    # Initialize data handler for training
    train_data_handler = DataHandler(api_url='http://api:5000')
    rl_trader = LearningTrader()

    lstm_predictor = LSTMPredictor(model_path='./lstm_model_v3_simple.h5', scaler_path='./scaler_v3_simple.pkl')

    # Define number of episodes and steps per episode
    num_episodes = 30
    steps_per_episode = 2000  # Adjusted to cover more data points per episode

    # Training environment
    train_env = TradingEnv(train_data_handler, lstm_predictor, rl_trader, steps_per_episode)

    # Initialize PPO model
    model = PPO('MlpPolicy', train_env, verbose=1)


    # Train the model
    train_rewards = train_model(train_env, model, num_episodes, steps_per_episode)

    # Save the trained model
    model.save("ppo_trading_model-2024-07-15")

    # Continuous trading loop
    while True:
        new_data = train_data_handler.get_current_data()
        if new_data is None:
            break
        state = train_env._get_observation(new_data)
        action, _ = model.predict(state)
        train_env.step(action)
        time.sleep(0.05)
