"""Random Forest model for traffic flow prediction."""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseTrafficModel


class RandomForestTrafficModel(BaseTrafficModel):
    """
    Random Forest regressor for traffic flow prediction.

    Flattens the sliding window input into a single feature vector
    so the tree-based model can work with it directly.
    """

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val, config=None):
        if config is None:
            config = {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5}

        # Flatten (samples, window, features) -> (samples, window*features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        self.model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 20),
            min_samples_split=config.get("min_samples_split", 5),
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train_flat, y_train)

        # Compute training score for history
        train_score = self.model.score(X_train_flat, y_train)
        history = {"train_r2": train_score}

        if X_val is not None and len(X_val) > 0:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            val_score = self.model.score(X_val_flat, y_val)
            history["val_r2"] = val_score

        return history

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def get_name(self):
        return "RandomForest"
