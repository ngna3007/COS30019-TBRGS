"""Abstract base class for traffic prediction models."""

from abc import ABC, abstractmethod
import numpy as np


class BaseTrafficModel(ABC):
    """Interface for all traffic prediction models."""

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, config=None):
        """
        Train the model.

        Args:
            X_train: training input sequences
            y_train: training targets
            X_val: validation input sequences
            y_val: validation targets
            config: model-specific config dict

        Returns:
            history dict with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions.

        Args:
            X: input sequences

        Returns:
            numpy array of predictions
        """
        pass

    @abstractmethod
    def save(self, path):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Load model from disk."""
        pass

    @abstractmethod
    def get_name(self):
        """Return model name string."""
        pass
