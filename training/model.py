from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class PricePredictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def save(self, path):
        """Save the model to disk"""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        """Load the model from disk"""
        loaded = joblib.load(path)
        instance = cls()
        instance.model = loaded
        instance.n_estimators = loaded.n_estimators
        instance.max_depth = loaded.max_depth
        return instance
