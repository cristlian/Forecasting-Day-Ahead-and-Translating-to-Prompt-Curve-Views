"""Test QA gate logic."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qa.checks import (
    CompletenessCheck,
    DuplicateCheck,
    RangeCheck,
    TemporalCheck,
    HourlyContinuityCheck,
    AlignmentCheck,
    run_all_checks,
    CheckResult,
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
                "forecast_load": {
                    "dtype": "float64",
                    "range": [20000, 90000],
                    "required": True,
                },
                "forecast_wind": {
                    "dtype": "float64",
                    "range": [0, 75000],
                    "required": True,
                },
                "forecast_solar": {
                    "dtype": "float64",
                    "range": [0, 55000],
                    "required": True,
                },
            }
        }
    }


def create_good_data():
    """Create a DataFrame that passes all QA checks."""
    dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    
    return pd.DataFrame({
        "day_ahead_price": np.random.uniform(30, 100, 48),
        "forecast_load": np.random.uniform(40000, 70000, 48),
        "forecast_wind": np.random.uniform(5000, 30000, 48),
        "forecast_solar": np.random.uniform(0, 20000, 48),
    }, index=dates)


def create_bad_data_missing():
    """Create a DataFrame with too much missing data."""
    dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    prices = np.random.uniform(30, 100, 48)
    # Make 30% missing (> 1% threshold)
    prices[10:25] = np.nan
    
    return pd.DataFrame({
        "day_ahead_price": prices,
        "forecast_load": np.random.uniform(40000, 70000, 48),
        "forecast_wind": np.random.uniform(5000, 30000, 48),
        "forecast_solar": np.random.uniform(0, 20000, 48),
    }, index=dates)


def create_bad_data_duplicates():
    """Create a DataFrame with duplicate timestamps."""
    dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    # Duplicate some timestamps
    dates_list = list(dates)
    dates_list[5] = dates_list[4]  # Create duplicate
    
    return pd.DataFrame({
        "day_ahead_price": np.random.uniform(30, 100, 48),
        "forecast_load": np.random.uniform(40000, 70000, 48),
        "forecast_wind": np.random.uniform(5000, 30000, 48),
        "forecast_solar": np.random.uniform(0, 20000, 48),
    }, index=dates_list)


def create_bad_data_gaps():
    """Create a DataFrame with large temporal gaps."""
    dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    dates_list = list(dates)
    # Create 5-hour gap
    for i in range(10, 15):
        dates_list[i] = dates_list[i] + pd.Timedelta(hours=5)
    
    df = pd.DataFrame({
        "day_ahead_price": np.random.uniform(30, 100, 48),
        "forecast_load": np.random.uniform(40000, 70000, 48),
        "forecast_wind": np.random.uniform(5000, 30000, 48),
        "forecast_solar": np.random.uniform(0, 20000, 48),
    }, index=dates_list)
    
    return df.sort_index()


class TestCompletenessCheck:
    """Tests for CompletenessCheck."""
    
    def test_passes_with_no_missing(self):
        """Test that check passes with no missing data."""
        config = create_test_config()
        df = create_good_data()
        
        check = CompletenessCheck()
        result = check.check(df, config)
        
        assert result.passed is True
        assert len(result.errors) == 0
    
    def test_fails_with_too_much_missing(self):
        """Test that check fails with too much missing data."""
        config = create_test_config()
        df = create_bad_data_missing()
        
        check = CompletenessCheck()
        result = check.check(df, config)
        
        assert result.passed is False
        assert len(result.errors) > 0
        assert "missing" in result.errors[0].lower()


class TestDuplicateCheck:
    """Tests for DuplicateCheck."""
    
    def test_passes_with_unique_timestamps(self):
        """Test that check passes with unique timestamps."""
        config = create_test_config()
        df = create_good_data()
        
        check = DuplicateCheck()
        result = check.check(df, config)
        
        assert result.passed is True
        assert len(result.errors) == 0
    
    def test_fails_with_duplicates(self):
        """Test that check fails with duplicate timestamps."""
        config = create_test_config()
        df = create_bad_data_duplicates()
        
        check = DuplicateCheck()
        result = check.check(df, config)
        
        assert result.passed is False
        assert len(result.errors) > 0


class TestTemporalCheck:
    """Tests for TemporalCheck."""
    
    def test_passes_with_continuous_data(self):
        """Test that check passes with continuous hourly data."""
        config = create_test_config()
        df = create_good_data()
        
        check = TemporalCheck()
        result = check.check(df, config)
        
        assert result.passed is True
        assert len(result.errors) == 0
    
    def test_fails_with_large_gaps(self):
        """Test that check fails with gaps larger than threshold."""
        config = create_test_config()
        df = create_bad_data_gaps()
        
        check = TemporalCheck()
        result = check.check(df, config)
        
        assert result.passed is False
        assert len(result.errors) > 0
        assert "gap" in result.errors[0].lower()


class TestQAGate:
    """Tests for QA gate evaluation."""
    
    def test_gate_passes_all_checks(self):
        """Test that gate passes when all checks pass."""
        results = [
            CheckResult(name="Check1", passed=True),
            CheckResult(name="Check2", passed=True),
            CheckResult(name="Check3", passed=True),
        ]
        
        assert evaluate_qa_gate(results) is True
    
    def test_gate_fails_on_any_failure(self):
        """Test that gate fails when any check fails."""
        results = [
            CheckResult(name="Check1", passed=True),
            CheckResult(name="Check2", passed=False, errors=["Error message"]),
            CheckResult(name="Check3", passed=True),
        ]
        
        with pytest.raises(QAGateFailure) as exc_info:
            evaluate_qa_gate(results)
        
        assert "Check2" in exc_info.value.failed_checks


class TestRunAllChecks:
    """Tests for run_all_checks integration."""
    
    def test_good_data_passes_all(self):
        """Test that good data passes all checks."""
        config = create_test_config()
        df = create_good_data()
        
        results = run_all_checks(df, config)
        
        # All checks should pass
        failed = [r for r in results if not r.passed]
        assert len(failed) == 0, f"Failed checks: {[r.name for r in failed]}"
    
    def test_bad_data_fails_expected_checks(self):
        """Test that bad data fails expected checks."""
        config = create_test_config()
        df = create_bad_data_missing()
        
        results = run_all_checks(df, config)
        
        # Should fail completeness check
        completeness_result = next(r for r in results if r.name == "Completeness Check")
        assert completeness_result.passed is False

