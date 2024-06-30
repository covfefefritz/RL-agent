from stable_baselines3 import PPO
from rl_env import TradingEnv
from predictor import LSTMPredictor
from data_handler import DataHandler
from rl_trader import LearningTrader
from keras.models import load_model
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


if __name__ == "__main__":
    data_handler = DataHandler(api_url='http://api:5000/get_data')
    rl_trader = LearningTrader()

    model = load_model('lstm_model_v3_simple.h5')
    # Save the model again
    model.save('lstm_model_v3_simple_new.h5')

    lstm_predictor = LSTMPredictor(model_path='./lstm_model_v3_simple_new.h5', scaler_path='./scaler_v3_simple.pkl')
    env = TradingEnv(data_handler, lstm_predictor, rl_trader)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    while True:
        new_data = data_handler.get_current_data()
        if new_data is None:
            break
        state = env._get_observation(new_data)
        action, _ = model.predict(state)
        env.step(action)
        time.sleep(0.05)