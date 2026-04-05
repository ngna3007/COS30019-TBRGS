"""GRU model for traffic flow prediction."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .base_model import BaseTrafficModel


class GRUTrafficModel(BaseTrafficModel):
    """GRU-based traffic flow predictor."""

    def __init__(self):
        self.model = None
        self.history = None

    def _build_model(self, input_shape, config):
        """Build the GRU architecture."""
        units = config.get("units", [64, 64])
        dropout = config.get("dropout", 0.2)
        lr = config.get("learning_rate", 0.001)

        model = Sequential()
        model.add(GRU(units[0], return_sequences=(len(units) > 1),
                      input_shape=input_shape))
        model.add(Dropout(dropout))

        for i in range(1, len(units)):
            return_seq = (i < len(units) - 1)
            model.add(GRU(units[i], return_sequences=return_seq))
            model.add(Dropout(dropout))

        model.add(Dense(1))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"]
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, config=None):
        if config is None:
            config = {"units": [64, 64], "dropout": 0.2, "epochs": 100,
                      "batch_size": 32, "learning_rate": 0.001, "patience": 10}

        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self._build_model(input_shape, config)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=config.get("patience", 10),
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get("epochs", 100),
            batch_size=config.get("batch_size", 32),
            callbacks=callbacks,
            verbose=0,
        )
        return self.history.history

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def get_name(self):
        return "GRU"
