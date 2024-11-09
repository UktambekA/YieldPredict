import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
import joblib

class YieldPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def create_stacking_model(self):
        """Create stacking ensemble model"""
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=62, max_depth=8, random_state=42)),
            ('br', BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8), 
                                  n_estimators=62, random_state=42)),
            ('lr', LinearRegression()),
            ('lasso', Lasso(alpha=0.001)),
            ('huber', HuberRegressor())
        ]
        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0)
        )
    
    def fit(self, X, y):
        """
        Fit the model
        
        Parameters:
        X (pandas.DataFrame): Training features
        y (pandas.Series): Target variable
        """
        self.feature_columns = X.columns
        self.model = self.create_stacking_model()
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        return mean_absolute_error(y, predictions)
    
    
    def save_model(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from disk"""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance