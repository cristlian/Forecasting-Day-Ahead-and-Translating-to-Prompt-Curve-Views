"""Test QA gate logic."""

import pytest
import pandas as pd
import numpy as np
from qa.checks import (
    CompletenessCheck,
    DuplicateCheck,
    RangeCheck,
    TemporalCheck,
    run_all_checks
)
from qa.gate import evaluate_qa_gate, QAGateFailure


def create_test_config():
    """Create a minimal test configuration."""
    return {
        "qa_thresholds": {
            "completeness": {
                "max_missing_pct": {
                    "critical_columns": 1.0,
                    "optional_columns": 10.0,
                }
            },
            "duplicates": {
                "max_duplicate_timestamps": 0
            },
            "ranges": {
                "fail_on_out_of_range": True,
                "warn_threshold_pct": 0.1,
            },
            "temporal": {
                "max_gap_hours": 2
            },
            "outliers": {
                "price_zscore_threshold": 10,
                "generation_zscore_threshold": 5,
            }
        },
        "schema": {
            "columns": {
                "day_ahead_price": {
                    "dtype": "float64",
                    "range": [-500, 3000],
                    "required": True,
                },
                "actual_load": {
                    "dtype": "float64",
                    "range": [0, 100000],
                    "required": True,
                }
            }
        }
    }


def test_completeness_check_passes():
    """Test that completeness check passes with no missing data."""
    config = create_test_config()
    
    df = pd.DataFrame({
        "day_ahead_price": [50.0, 60.0, 70.0],
        "actual_load": [5000.0, 6000.0, 7000.0],
    })
    
    check = CompletenessCheck()
    result = check.check(df, config)
    
    assert result is True
    assert len(check.errors) == 0


def test_completeness_check_fails():
    """Test that completeness check fails with too much missing data."""
    config = create_test_config()
    
    df = pd.DataFrame({
        "day_ahead_price": [50.0, np.nan, np.nan],  # 66% missing > 1% threshold
        "actual_load": [5000.0, 6000.0, 7000.0],
    })
    
    check = CompletenessCheck()
    result = check.check(df, config)
    
    assert result is False
    assert len(check.errors) > 0


def test_duplicate_check_passes():
    """Test that duplicate check passes with unique timestamps."""
    config = create_test_config()
    
    dates = pd.date_range("2024-01-01", periods=3, freq="H")
    df = pd.DataFrame({
        "day_ahead_price": [50.0, 60.0, 70.0],
    }, index=dates)
    
    check = DuplicateCheck()
    result = check.check(df, config)
    
    assert result is True


def test_duplicate_check_fails():
    """Test that duplicate check fails with duplicate timestamps."""
    config = create_test_config()
    
    # Create duplicate timestamps
    dates = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00"])
    df = pd.DataFrame({
        "day_ahead_price": [50.0, 60.0, 70.0],
    }, index=dates)
    
    check = DuplicateCheck()
    result = check.check(df, config)
    
    assert result is False
    assert len(check.errors) > 0


def test_range_check_passes():
    """Test that range check passes with values in range."""
    config = create_test_config()
    
    df = pd.DataFrame({
        "day_ahead_price": [50.0, 60.0, 70.0],  # within [-500, 3000]
        "actual_load": [5000.0, 6000.0, 7000.0],  # within [0, 100000]
    })
    
    check = RangeCheck()
    result = check.check(df, config)
    
    assert result is True


def test_range_check_fails():
    """Test that range check fails with out-of-range values."""
    config = create_test_config()
    
    df = pd.DataFrame({
        "day_ahead_price": [50.0, 5000.0, 70.0],  # 5000 > 3000 (out of range)
        "actual_load": [5000.0, 6000.0, 7000.0],
    })
    
    check = RangeCheck()
    result = check.check(df, config)
    
    assert result is False


def test_qa_gate_evaluates_correctly():
    """Test that QA gate evaluates all checks correctly."""
    config = create_test_config()
    
    # Good data
    df_good = pd.DataFrame({
        "day_ahead_price": [50.0, 60.0, 70.0],
        "actual_load": [5000.0, 6000.0, 7000.0],
    })
    
    checks = run_all_checks(df_good, config)
    
    # Should pass
    result = evaluate_qa_gate(checks)
    assert result is True


def test_qa_gate_fails_with_bad_data():
    """Test that QA gate fails with bad data."""
    config = create_test_config()
    
    # Bad data (too much missing)
    df_bad = pd.DataFrame({
        "day_ahead_price": [50.0, np.nan, np.nan],
        "actual_load": [5000.0, 6000.0, 7000.0],
    })
    
    checks = run_all_checks(df_bad, config)
    
    # Should fail
    with pytest.raises(QAGateFailure):
        evaluate_qa_gate(checks)
