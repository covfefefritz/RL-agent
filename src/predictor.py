import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging
from utils import add_time_features, create_sequences

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class LSTMPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.seq_length = 60
        self.data = []
        self.predictions = []  # Initialize as an empty list
        logger.info("LSTMPredictor initialized with model_path: %s and scaler_path: %s", model_path, scaler_path)

    def add_data(self, new_data):
        self.data.append(new_data)

    def preprocess_data(self):
        if len(self.data) < self.seq_length + 1:
            return None

        df = pd.DataFrame(self.data)
        try:
            df['Gmt time'] = pd.to_datetime(df['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
        except ValueError as e:
            logging.error(f"Error parsing datetime: {e}")
            return None
        df.index = df['Gmt time']  # Set 'Gmt time' as the index

        df = add_time_features(df)
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

        last_sequence = X_new[-1]
        recursive_predictions = self.recursive_predict(last_sequence, steps=15)

        self.predictions = list(recursive_predictions)  # Ensure predictions is a list

    def recursive_predict(self, sequence, steps=1):
        sequence = sequence.copy()
        predictions = []

        for step in range(steps):
            prediction = self.model.predict(sequence[np.newaxis, :, :])[0, 0]
            predictions.append(prediction)
            sequence = np.roll(sequence, -1, axis=0)
            sequence[-1, 3] = prediction

        prediction_scaled = np.zeros((len(predictions), sequence.shape[1]))
        prediction_scaled[:, 3] = predictions
        inverse_transformed = self.scaler.inverse_transform(prediction_scaled)

        return inverse_transformed[:, 3]
