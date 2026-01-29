"""Test feature engineering to ensure no data leakage."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.build import (
    build_features,
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    add_derived_features,
    validate_no_leakage,
    get_feature_columns,
)


def create_test_config():
    """Create a test configuration for feature engineering."""
    return {
        "market": {
            "target": {"column": "day_ahead_price"}
        },
        "features": {
            "feature_toggles": {
                "enable_calendar": True,
                "enable_lags": True,
                "enable_rolling": True,
                "enable_residual_load": True,
                "enable_interactions": True,
            },
            "calendar_features": ["hour", "day_of_week", "month", "is_weekend"],
            "lag_features": {
                "columns": ["day_ahead_price", "forecast_load"],
                "lags": [24, 48, 168]
            },
            "rolling_features": {
                "columns": ["day_ahead_price"],
                "windows": [24, 168],
                "aggregations": ["mean", "std"],
                "min_periods": 12
            },
            "derived_features": {
                "forecast_residual_load": {
                    "formula": "forecast_load - forecast_wind - forecast_solar"
                }
            },
            "interaction_features": [["hour", "day_of_week"]]
        }
    }


def create_test_data(hours=500):
    """Create test data for feature engineering."""
    dates = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    
    return pd.DataFrame({
        "day_ahead_price": np.random.uniform(30, 100, hours),
        "forecast_load": np.random.uniform(40000, 70000, hours),
        "forecast_wind": np.random.uniform(5000, 30000, hours),
        "forecast_solar": np.random.uniform(0, 20000, hours),
    }, index=dates)


class TestLagFeatures:
    """Tests for lag feature generation."""
    
    def test_lag_features_correct_values(self):
        """Test that lag features have correct values."""
        dates = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        df = pd.DataFrame({
            "day_ahead_price": np.arange(200).astype(float),
        }, index=dates)
        
        config = {
            "columns": ["day_ahead_price"],
            "lags": [24, 48]
        }
        
        df_lagged = add_lag_features(df.copy(), config)
        
        # Lag-24: value at hour 24 should equal original value at hour 0
        assert df_lagged.loc[dates[24], "lag_24h_day_ahead_price"] == 0.0
        assert df_lagged.loc[dates[48], "lag_24h_day_ahead_price"] == 24.0
        
        # Lag-48: value at hour 48 should equal original value at hour 0
        assert df_lagged.loc[dates[48], "lag_48h_day_ahead_price"] == 0.0
    
    def test_lag_features_have_nan_at_start(self):
        """Test that lag features have NaN for initial rows."""
        df = create_test_data(100)
        
        config = {"columns": ["day_ahead_price"], "lags": [24, 48]}
        df_lagged = add_lag_features(df.copy(), config)
        
        # First 24 rows should have NaN for lag-24
        assert df_lagged["lag_24h_day_ahead_price"].iloc[:24].isna().all()
        
        # First 48 rows should have NaN for lag-48
        assert df_lagged["lag_48h_day_ahead_price"].iloc[:48].isna().all()
    
    def test_lag_features_no_future_leakage(self):
        """Test that lag features don't use future data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        df = pd.DataFrame({
            "day_ahead_price": np.arange(100).astype(float),
        }, index=dates)
        
        config = {"columns": ["day_ahead_price"], "lags": [24]}
        df_lagged = add_lag_features(df.copy(), config)
        
        # For any row i, lag-24 should never equal the current value (unless by chance)
        # More importantly, lag should always be from PAST, not future
        for i in range(24, 100):
            lag_val = df_lagged["lag_24h_day_ahead_price"].iloc[i]
            current_val = df_lagged["day_ahead_price"].iloc[i]
            # Lag value should be from 24 hours ago
            assert lag_val == current_val - 24, f"Lag mismatch at row {i}"


class TestRollingFeatures:
    """Tests for rolling feature generation."""
    
    def test_rolling_mean_correct(self):
        """Test that rolling mean is computed correctly."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        df = pd.DataFrame({
            "day_ahead_price": np.arange(100).astype(float),
        }, index=dates)
        
        config = {
            "columns": ["day_ahead_price"],
            "windows": [24],
            "aggregations": ["mean"],
            "min_periods": 12
        }
        
        df_rolling = add_rolling_features(df.copy(), config)
        
        # Rolling features are computed then shifted by 24 to avoid leakage
        # At index 48: rolling(0:47).mean() shifted by 24 = rolling(0:23).mean()
        # But with min_periods=12, the first value with valid rolling appears at index 35
        # which is shift(24) from when window first fills (index 11 with min_periods=12)
        # Check that rolling feature has reasonable value
        actual_mean = df_rolling["roll_24h_mean_day_ahead_price"].iloc[48]
        # Mean of indices 1-24 (shifted by 24 from indices 25-48)
        expected_mean = np.mean(np.arange(1, 25))
        assert np.isclose(actual_mean, expected_mean), f"Expected {expected_mean}, got {actual_mean}"
    
    def test_rolling_features_no_leakage(self):
        """Test that rolling features don't leak future data."""
        df = create_test_data(200)
        
        config = {
            "columns": ["day_ahead_price"],
            "windows": [24],
            "aggregations": ["mean"],
            "min_periods": 12
        }
        
        df_rolling = add_rolling_features(df.copy(), config)
        
        # Rolling features should have NaN for early rows
        # With min_periods=12, rolling starts producing values at index 11
        # After shift(24), first valid value is at index 35
        # So first 35 rows should be NaN
        assert df_rolling["roll_24h_mean_day_ahead_price"].iloc[:35].isna().all()


class TestCalendarFeatures:
    """Tests for calendar feature extraction."""
    
    def test_hour_extraction(self):
        """Test that hour is extracted correctly."""
        dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(index=dates)
        
        df_cal = add_calendar_features(df.copy(), ["hour"])
        
        assert df_cal["hour"].iloc[0] == 0
        assert df_cal["hour"].iloc[12] == 12
        assert df_cal["hour"].iloc[23] == 23
    
    def test_day_of_week_extraction(self):
        """Test that day of week is extracted correctly."""
        # 2024-01-01 was a Monday
        dates = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
        df = pd.DataFrame(index=dates)
        
        df_cal = add_calendar_features(df.copy(), ["day_of_week"])
        
        # Monday = 0
        assert df_cal["day_of_week"].iloc[0] == 0
        # Tuesday = 1 (24 hours later)
        assert df_cal["day_of_week"].iloc[24] == 1
        # Sunday = 6
        assert df_cal["day_of_week"].iloc[144] == 6
    
    def test_is_weekend(self):
        """Test that weekend flag is correct."""
        dates = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
        df = pd.DataFrame(index=dates)
        
        df_cal = add_calendar_features(df.copy(), ["is_weekend"])
        
        # Monday through Friday: is_weekend = 0
        assert df_cal["is_weekend"].iloc[0] == 0  # Monday
        
        # Saturday and Sunday: is_weekend = 1
        # Saturday 2024-01-06 starts at hour 120 (5 * 24)
        assert df_cal["is_weekend"].iloc[120] == 1


class TestDerivedFeatures:
    """Tests for derived feature computation."""
    
    def test_residual_load_correct(self):
        """Test that residual load is computed correctly."""
        df = pd.DataFrame({
            "forecast_load": [50000.0, 60000.0, 70000.0],
            "forecast_wind": [10000.0, 15000.0, 20000.0],
            "forecast_solar": [5000.0, 10000.0, 15000.0],
        })
        
        config = {
            "forecast_residual_load": {
                "formula": "forecast_load - forecast_wind - forecast_solar"
            }
        }
        
        df_derived = add_derived_features(df.copy(), config)
        
        # Residual = Load - Wind - Solar
        expected = [35000.0, 35000.0, 35000.0]
        actual = df_derived["forecast_residual_load"].tolist()
        
        assert actual == expected


class TestBuildFeatures:
    """Tests for full feature pipeline."""
    
    def test_build_features_end_to_end(self):
        """Test full feature pipeline."""
        config = create_test_config()
        df = create_test_data(500)
        
        df_features = build_features(df, config)
        
        # Should have more columns than input
        assert len(df_features.columns) > len(df.columns)
        
        # Should have calendar features
        assert "hour" in df_features.columns
        assert "day_of_week" in df_features.columns
        
        # Should have lag features
        assert "lag_24h_day_ahead_price" in df_features.columns
        
        # Should have rolling features
        assert "roll_24h_mean_day_ahead_price" in df_features.columns
    
    def test_validate_no_leakage_passes(self):
        """Test leakage validation passes for correct config."""
        config = create_test_config()
        df = create_test_data(500)
        
        df_features = build_features(df, config)
        
        assert validate_no_leakage(df_features, config) is True
    
    def test_get_feature_columns(self):
        """Test that feature column extraction works."""
        config = create_test_config()
        df = create_test_data(500)
        
        df_features = build_features(df, config)
        feature_cols = get_feature_columns(df_features, config)
        
        # Should not include target
        assert "day_ahead_price" not in feature_cols
        
        # Should include lag features
        lag_cols = [c for c in feature_cols if c.startswith("lag_")]
        assert len(lag_cols) > 0
