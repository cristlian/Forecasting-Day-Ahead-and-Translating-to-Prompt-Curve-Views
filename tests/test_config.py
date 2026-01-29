"""Test configuration loading and validation."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.config import load_config, load_yaml


@pytest.fixture
def config():
    """Load configuration for tests."""
    try:
        return load_config("config")
    except FileNotFoundError:
        pytest.skip("Config directory not found")


def test_load_yaml():
    """Test loading a YAML file."""
    assert callable(load_yaml)


def test_load_config_structure(config):
    """Test that config has required top-level keys."""
    required_keys = [
        "market",
        "schema",
        "qa_thresholds",
        "features",
    ]
    
    for key in required_keys:
        assert key in config, f"Missing required config key: {key}"


def test_market_config_de_lu(config):
    """Test that market config is set to DE-LU."""
    # market.yaml has nested structure: config["market"]["market"]
    market_yaml = config["market"]
    market_config = market_yaml.get("market", market_yaml)
    
    # Check market code
    assert "code" in market_config, "Missing market code"
    assert market_config["code"] == "DE_LU", "Market should be DE_LU (Germany-Luxembourg)"
    
    # Check timezone
    assert "timezone" in market_config
    assert market_config["timezone"] == "Europe/Berlin"
    


def test_market_has_target(config):
    """Test that market config has target definition."""
    assert "target" in config["market"]
    target = config["market"]["target"]
    
    assert "column" in target
    assert target["column"] == "day_ahead_price"
    assert "unit" in target
    assert target["unit"] == "EUR/MWh"


def test_market_has_drivers(config):
    """Test that market config has fundamental drivers."""
    assert "drivers" in config["market"]
    drivers = config["market"]["drivers"]
    
    assert len(drivers) >= 3, "Should have at least 3 drivers (load, wind, solar)"
    
    driver_names = [d["name"] for d in drivers]
    assert "forecast_load" in driver_names
    assert "forecast_wind" in driver_names
    assert "forecast_solar" in driver_names


def test_schema_columns_exist(config):
    """Test that schema defines required columns."""
    schema = config["schema"]
    
    assert "columns" in schema
    assert len(schema["columns"]) > 0
    
    # Check required columns exist
    required_cols = ["timestamp", "day_ahead_price", "forecast_load", "forecast_wind", "forecast_solar"]
    for col in required_cols:
        assert col in schema["columns"], f"Missing schema column: {col}"


def test_schema_columns_have_ranges(config):
    """Test that numeric columns have valid ranges."""
    schema = config["schema"]
    
    for col_name, col_def in schema["columns"].items():
        if col_def.get("dtype") == "float64" and "range" in col_def:
            min_val, max_val = col_def["range"]
            assert min_val < max_val, f"Invalid range for {col_name}"


def test_qa_thresholds_exist(config):
    """Test that QA thresholds are defined."""
    qa_config = config["qa_thresholds"]
    
    required_sections = ["completeness", "duplicates", "ranges", "temporal", "outliers"]
    
    for section in required_sections:
        assert section in qa_config, f"Missing QA section: {section}"


def test_qa_completeness_thresholds(config):
    """Test that completeness thresholds are reasonable."""
    completeness = config["qa_thresholds"]["completeness"]
    
    assert "max_missing_pct" in completeness
    critical = completeness["max_missing_pct"]["critical_columns"]
    optional = completeness["max_missing_pct"]["optional_columns"]
    
    assert 0 <= critical <= 5, "Critical threshold should be 0-5%"
    assert 0 <= optional <= 20, "Optional threshold should be 0-20%"


def test_features_config_valid(config):
    """Test that features config is valid."""
    features = config["features"]
    
    assert "feature_toggles" in features
    assert "lag_features" in features
    assert "rolling_features" in features
    
    # Check lags are >= 24 (no leakage for D+1 prediction)
    lags = features["lag_features"].get("lags", [])
    assert all(lag >= 24 for lag in lags), "All lags must be >= 24h for D+1 prediction"


def test_features_lag_columns_are_forecasts(config):
    """Test that lag features use forecasts, not actuals."""
    lag_config = config["features"]["lag_features"]
    
    columns = lag_config.get("columns", [])
    
    # Should not include actuals
    for col in columns:
        assert "actual" not in col.lower(), f"Lag column {col} appears to be actual, not forecast"
