from stable_baselines3 import PPO
from keras.models import load_model
import numpy as np
import time
import logging

from rl_env import TradingEnv  # Ensure correct import
from predictor import LSTMPredictor
from data_handler import DataHandler
from rl_trader import LearningTrader

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
    return rewards

if __name__ == "__main__":
    # Initialize data handler for training
    train_data_handler = DataHandler(api_url='http://api:5000')
    rl_trader = LearningTrader()

    model = load_model('lstm_model_v3_simple.h5')
    lstm_predictor = LSTMPredictor(model_path='./lstm_model_v3_simple.h5', scaler_path='./scaler_v3_simple.pkl')

    # Training environment
    train_env = TradingEnv(train_data_handler, lstm_predictor, rl_trader)

    # Initialize PPO model
    model = PPO('MlpPolicy', train_env, verbose=1)

    # Define number of episodes and steps per episode
    num_episodes = 35
    steps_per_episode = 2000  # Adjusted to cover more data points per episode

    # Train the model
    train_rewards = train_model(train_env, model, num_episodes, steps_per_episode)

    # Save the trained model
    model.save("ppo_trading_model-2024-07-10")

    # Continuous trading loop
    while True:
        new_data = train_data_handler.get_current_data()
        if new_data is None:
            break
        state = train_env._get_observation(new_data)
        action, _ = model.predict(state)
        train_env.step(action)
        time.sleep(0.05)
