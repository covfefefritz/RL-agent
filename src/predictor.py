import numpy as np
import joblib
from utils import create_sequences
from keras.models import load_model
import tensorflow as tf


class LSTMPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.seq_length = 60

    def predict(self, data, seq_length):
        if len(data) < seq_length:
            return None

        df_scaled = self.scaler.transform(data.values)
        X_new, _ = create_sequences(df_scaled, seq_length)

        if len(X_new) == 0:
            return None

        last_sequence = X_new[-1]
        predictions = self.recursive_predict(last_sequence, steps=15)
        return predictions[-1]

    def recursive_predict(self, sequence, steps=1):
        sequence = sequence.copy()
        predictions = []

        for step in range(steps):
            prediction = self.model.predict(sequence[np.newaxis, :, :])[0, 0]
            predictions.append(prediction)
            sequence = np.roll(sequence, -1, axis=0)
            sequence[-1, 3] = prediction  # Assuming 'Close' is the 4th feature (index 3)

        prediction_scaled = np.zeros((len(predictions), sequence.shape[1]))
        prediction_scaled[:, 3] = predictions
        inverse_transformed = self.scaler.inverse_transform(prediction_scaled)
        return inverse_transformed[:, 3]
