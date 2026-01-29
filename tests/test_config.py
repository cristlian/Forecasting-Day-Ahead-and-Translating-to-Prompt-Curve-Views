"""Test configuration loading and validation."""

import pytest
from pathlib import Path
from pipeline.config import load_config, load_yaml


def test_load_yaml():
    """Test loading a YAML file."""
    # This would require a test config file
    # For now, just test the function exists
    assert callable(load_yaml)


def test_load_config_structure():
    """Test that config has required top-level keys."""
    try:
        config = load_config("config")
        
        required_keys = [
            "market",
            "schema",
            "qa_thresholds",
            "features",
            "model",
            "reporting"
        ]
        
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"
    
    except FileNotFoundError:
        pytest.skip("Config directory not found")


def test_market_config_keys():
    """Test that market config has required keys."""
    try:
        config = load_config("config")
        market_config = config["market"]
        
        required_keys = ["market", "date_range", "target"]
        
        for key in required_keys:
            assert key in market_config, f"Missing market config key: {key}"
    
    except FileNotFoundError:
        pytest.skip("Config directory not found")


def test_schema_columns_exist():
    """Test that schema defines columns."""
    try:
        config = load_config("config")
        schema = config["schema"]
        
        assert "columns" in schema
        assert len(schema["columns"]) > 0
        
        # Check that target column is in schema
        target_col = config["market"]["target"]["column"]
        assert target_col in schema["columns"]
    
    except FileNotFoundError:
        pytest.skip("Config directory not found")


def test_qa_thresholds_exist():
    """Test that QA thresholds are defined."""
    try:
        config = load_config("config")
        qa_config = config["qa_thresholds"]
        
        required_sections = ["completeness", "duplicates", "ranges", "temporal"]
        
        for section in required_sections:
            assert section in qa_config
    
    except FileNotFoundError:
        pytest.skip("Config directory not found")
