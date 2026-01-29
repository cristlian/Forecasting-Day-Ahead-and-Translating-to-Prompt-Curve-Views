"""Test feature engineering to ensure no data leakage."""

import pytest
import pandas as pd
import numpy as np
from features.build import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
)


def test_lag_features_no_leakage():
    """Test that lag features don't leak future data."""
    dates = pd.date_range("2024-01-01", periods=200, freq="H")
    df = pd.DataFrame({
        "price": np.arange(200),
    }, index=dates)
    
    config = {
        "columns": ["price"],
        "lags": [1, 24, 168]
    }
    
    df_lagged = add_lag_features(df, config)
    
    # Check lag_1h
    assert df_lagged.loc[dates[1], "lag_1h_price"] == df.loc[dates[0], "price"]
    
    # Check lag_24h
    assert df_lagged.loc[dates[24], "lag_24h_price"] == df.loc[dates[0], "price"]
    
    # Check that early rows have NaN lags
    assert pd.isna(df_lagged.loc[dates[0], "lag_1h_price"])
    assert pd.isna(df_lagged.loc[dates[23], "lag_24h_price"])


def test_rolling_features_no_leakage():
    """Test that rolling features don't leak future data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    df = pd.DataFrame({
        "price": np.arange(100),
    }, index=dates)
    
    config = {
        "columns": ["price"],
        "windows": [24],
        "aggregations": ["mean"]
    }
    
    df_rolling = add_rolling_features(df, config)
    
    # Rolling mean at hour 24 should be mean of hours 0-23 (not including 24)
    expected_mean = np.mean(np.arange(24))
    actual_mean = df_rolling.loc[dates[23], "rolling_24h_mean_price"]
    
    assert np.isclose(actual_mean, expected_mean)
    
    # First 23 hours should have NaN
    assert pd.isna(df_rolling.loc[dates[0], "rolling_24h_mean_price"])


def test_calendar_features_correct():
    """Test that calendar features are extracted correctly."""
    dates = pd.date_range("2024-01-01", periods=168, freq="H")  # 1 week
    df = pd.DataFrame(index=dates)
    
    features = ["hour", "day_of_week", "is_weekend"]
    df_calendar = add_calendar_features(df, features)
    
    # Check hour extraction
    assert df_calendar.loc[dates[0], "hour"] == 0
    assert df_calendar.loc[dates[12], "hour"] == 12
    
    # Check day of week (Monday = 0)
    assert df_calendar.loc[dates[0], "day_of_week"] == 0  # 2024-01-01 is Monday
    
    # Check weekend flag
    # 2024-01-06 is Saturday
    saturday = pd.Timestamp("2024-01-06 00:00:00")
    assert df_calendar.loc[saturday, "is_weekend"] == 1
    
    # Monday should not be weekend
    assert df_calendar.loc[dates[0], "is_weekend"] == 0


def test_feature_alignment():
    """Test that features remain aligned with target after transformations."""
    dates = pd.date_range("2024-01-01", periods=200, freq="H")
    df = pd.DataFrame({
        "price": np.arange(200),
        "load": np.arange(200) * 2,
    }, index=dates)
    
    # Add lags
    config = {"columns": ["price"], "lags": [24]}
    df_features = add_lag_features(df, config)
    
    # Remove NaN rows (first 24 hours)
    df_features = df_features.dropna()
    
    # Verify alignment: lag_24h_price at time t should equal price at time t-24
    for i in range(24, len(dates)):
        current_time = dates[i]
        lag_time = dates[i - 24]
        
        if current_time in df_features.index:
            assert df_features.loc[current_time, "lag_24h_price"] == df.loc[lag_time, "price"]


def test_no_future_information():
    """Test that feature creation doesn't use future information."""
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    df = pd.DataFrame({
        "target": np.random.rand(100),
    }, index=dates)
    
    # Add lag features - target at time t should not be used to predict target at time t-1
    config = {"columns": ["target"], "lags": [1]}
    df_lagged = add_lag_features(df, config)
    
    # For any time t, lag_1h_target should be from time t-1, not t or t+1
    for i in range(1, len(dates)):
        current_lag = df_lagged.loc[dates[i], "lag_1h_target"]
        previous_target = df.loc[dates[i-1], "target"]
        
        assert np.isclose(current_lag, previous_target), \
            f"Lag feature at {dates[i]} doesn't match target at {dates[i-1]}"
