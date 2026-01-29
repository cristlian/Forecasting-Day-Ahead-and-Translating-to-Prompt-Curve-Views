"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """
    Load all configuration files.
    
    Args:
        config_dir: Path to configuration directory
    
    Returns:
        Dictionary containing all configuration
    """
    config_path = Path(config_dir)
    
    config = {
        "market": load_yaml(config_path / "market.yaml"),
        "schema": load_yaml(config_path / "schema.yaml"),
        "qa_thresholds": load_yaml(config_path / "qa_thresholds.yaml"),
        "features": load_yaml(config_path / "features.yaml"),
        "model": load_yaml(config_path / "model.yaml"),
        "reporting": load_yaml(config_path / "reporting.yaml"),
    }
    
    # Validate required keys
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that all required configuration keys are present.
    
    Raises:
        ValueError: If required keys are missing
    """
    # Check market config
    required_market = ["code", "timezone"]
    for key in required_market:
        if key not in config["market"]["market"]:
            raise ValueError(f"Missing required market config key: {key}")
    
    # Check target config
    if "target" not in config["market"]:
        raise ValueError("Missing target configuration")
    
    # Check schema columns
    if "columns" not in config["schema"]:
        raise ValueError("Missing schema columns definition")
    
    # Check QA thresholds
    if "completeness" not in config["qa_thresholds"]:
        raise ValueError("Missing QA completeness thresholds")
    
    # Check model config
    if "improved_model" not in config["model"]:
        raise ValueError("Missing improved_model configuration")
