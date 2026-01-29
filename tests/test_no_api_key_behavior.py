"""Tests for missing API key behavior in validation."""

import os
import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set matplotlib backend to non-GUI before any imports that might use it
import matplotlib
matplotlib.use("Agg")


def _clear_env(monkeypatch):
    for var in ["ENTSOE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)


def test_validation_without_api_keys(monkeypatch):
    from pipeline.config import load_config
    from validation.runner import run_validation

    _clear_env(monkeypatch)

    config = load_config(Path(__file__).parent.parent / "config")
    result = run_validation(
        config=config,
        validation_date="2026-01-29",
        use_sample=True,
        cache_only=False,
        llm_test=True,
    )

    assert result.report_path.exists()
    assert result.metrics_path.exists()
    assert result.llm_test["ran"] is True
    assert "api key" in result.llm_test["status"].lower()


def test_validation_cache_only_missing_cache(monkeypatch):
    """
    Test that validation with cache_only=True works if local raw data is available.
    
    Previously this tested that CacheMissingError was raised, but now
    the system gracefully falls back to local raw data even when cache_only=True.
    """
    from pipeline.config import load_config
    from validation.runner import run_validation

    _clear_env(monkeypatch)

    config = load_config(Path(__file__).parent.parent / "config")

    # With local raw data available, validation should succeed even without cache
    # We use a date within the data range
    result = run_validation(
        config=config,
        validation_date="2024-09-01",
        use_sample=False,
        cache_only=True,
        llm_test=False,
    )
    
    # Should succeed using local raw data
    assert result.report_path.exists()
    assert result.metrics_path.exists()
