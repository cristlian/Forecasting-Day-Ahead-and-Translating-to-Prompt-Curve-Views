"""Reusable feature transformers."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ResidualLoadTransformer(BaseEstimator, TransformerMixin):
    """Compute residual load from load and renewable generation."""
    
    def __init__(self):
        self.load_col = "actual_load"
        self.wind_col = "wind_generation"
        self.solar_col = "solar_generation"
    
    def fit(self, X, y=None):
        """Fit (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform by computing residual load."""
        X = X.copy()
        X["residual_load"] = (
            X[self.load_col] - X[self.wind_col] - X[self.solar_col]
        )
        return X


class CalendarFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract calendar features from datetime index."""
    
    def __init__(self, features=None):
        if features is None:
            features = ["hour", "day_of_week", "month", "is_weekend"]
        self.features = features
    
    def fit(self, X, y=None):
        """Fit (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform by adding calendar features."""
        X = X.copy()
        
        if "hour" in self.features:
            X["hour"] = X.index.hour
        
        if "day_of_week" in self.features:
            X["day_of_week"] = X.index.dayofweek
        
        if "month" in self.features:
            X["month"] = X.index.month
        
        if "is_weekend" in self.features:
            X["is_weekend"] = (X.index.dayofweek >= 5).astype(int)
        
        return X


class LagTransformer(BaseEstimator, TransformerMixin):
    """Add lagged features."""
    
    def __init__(self, columns, lags):
        self.columns = columns
        self.lags = lags
    
    def fit(self, X, y=None):
        """Fit (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform by adding lag features."""
        X = X.copy()
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            for lag in self.lags:
                X[f"lag_{lag}h_{col}"] = X[col].shift(lag)
        
        return X


class RollingTransformer(BaseEstimator, TransformerMixin):
    """Add rolling window features."""
    
    def __init__(self, columns, windows, aggregations=None):
        self.columns = columns
        self.windows = windows
        if aggregations is None:
            aggregations = ["mean", "std"]
        self.aggregations = aggregations
    
    def fit(self, X, y=None):
        """Fit (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform by adding rolling features."""
        X = X.copy()
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            for window in self.windows:
                for agg in self.aggregations:
                    feature_name = f"rolling_{window}h_{agg}_{col}"
                    
                    if agg == "mean":
                        X[feature_name] = X[col].rolling(window=window).mean()
                    elif agg == "std":
                        X[feature_name] = X[col].rolling(window=window).std()
                    elif agg == "min":
                        X[feature_name] = X[col].rolling(window=window).min()
                    elif agg == "max":
                        X[feature_name] = X[col].rolling(window=window).max()
        
        return X
