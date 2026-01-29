"""Test schema enforcement and validation."""

import pytest
import pandas as pd
import numpy as np
from utils.io import _enforce_schema, _validate_schema


def test_enforce_schema_dtypes():
    """Test that schema enforcement converts dtypes correctly."""
    schema = {
        "columns": {
            "price": {"dtype": "float64", "required": True},
            "load": {"dtype": "float64", "required": True},
        }
    }
    
    df = pd.DataFrame({
        "price": ["100.5", "200.3"],
        "load": [5000, 6000],
    })
    
    df_enforced = _enforce_schema(df, schema)
    
    assert df_enforced["price"].dtype == np.float64
    assert df_enforced["load"].dtype == np.float64


def test_schema_validation_required_columns():
    """Test that validation catches missing required columns."""
    schema = {
        "columns": {
            "price": {"dtype": "float64", "required": True},
            "load": {"dtype": "float64", "required": True},
        }
    }
    
    df = pd.DataFrame({
        "price": [100.5, 200.3],
        # missing "load"
    })
    
    with pytest.raises(ValueError, match="Required column"):
        _validate_schema(df, schema)


def test_schema_validation_passes():
    """Test that validation passes with valid data."""
    schema = {
        "columns": {
            "price": {"dtype": "float64", "required": True},
            "load": {"dtype": "float64", "required": True},
        }
    }
    
    df = pd.DataFrame({
        "price": [100.5, 200.3],
        "load": [5000.0, 6000.0],
    })
    
    # Should not raise
    _validate_schema(df, schema)


def test_datetime_index_timezone_aware():
    """Test that datetime columns can be made timezone-aware."""
    dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="Europe/Berlin")
    df = pd.DataFrame({"price": np.random.rand(24)}, index=dates)
    
    assert df.index.tz is not None
    assert str(df.index.tz) == "Europe/Berlin"
