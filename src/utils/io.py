"""File I/O utilities with schema enforcement."""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def read_with_schema(
    filepath: Path,
    schema: Optional[Dict] = None,
    enforce_types: bool = True
) -> pd.DataFrame:
    """
    Read data file with schema enforcement.
    
    Args:
        filepath: Path to data file (parquet or CSV)
        schema: Optional schema dictionary
        enforce_types: Whether to enforce data types
    
    Returns:
        DataFrame with validated schema
    """
    logger.info(f"Reading {filepath}")
    
    # Determine file type and read
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    # Enforce schema if provided
    if schema and enforce_types:
        df = _enforce_schema(df, schema)
    
    return df


def write_with_schema(
    df: pd.DataFrame,
    filepath: Path,
    schema: Optional[Dict] = None,
    format: str = 'parquet'
) -> None:
    """
    Write data file with schema validation.
    
    Args:
        df: DataFrame to write
        filepath: Path to save file
        schema: Optional schema dictionary for validation
        format: Output format ('parquet' or 'csv')
    """
    # Validate schema if provided
    if schema:
        _validate_schema(df, schema)
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    if format == 'parquet':
        df.to_parquet(filepath)
    elif format == 'csv':
        df.to_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Wrote {len(df)} rows to {filepath}")


def _enforce_schema(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """Enforce schema types on dataframe."""
    df = df.copy()
    
    for col, spec in schema["columns"].items():
        if col not in df.columns:
            if spec.get("required", False):
                raise ValueError(f"Required column '{col}' not found in data")
            continue
        
        # Convert dtype
        dtype = spec["dtype"]
        if dtype.startswith("datetime"):
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)
    
    return df


def _validate_schema(df: pd.DataFrame, schema: Dict) -> None:
    """Validate dataframe against schema."""
    # Check required columns
    for col, spec in schema["columns"].items():
        if spec.get("required", False) and col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from data")
    
    # Check data types
    for col in df.columns:
        if col in schema["columns"]:
            expected_dtype = schema["columns"][col]["dtype"]
            actual_dtype = str(df[col].dtype)
            
            # Simplified type checking
            if "int" in expected_dtype and "int" not in actual_dtype:
                logger.warning(f"Column '{col}' dtype mismatch: expected {expected_dtype}, got {actual_dtype}")
            elif "float" in expected_dtype and "float" not in actual_dtype:
                logger.warning(f"Column '{col}' dtype mismatch: expected {expected_dtype}, got {actual_dtype}")


def read_config_safe(filepath: Path) -> Dict:
    """
    Read YAML config file with error handling.
    
    Args:
        filepath: Path to YAML file
    
    Returns:
        Configuration dictionary
    """
    import yaml
    
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to read config from {filepath}: {e}")
        raise


def save_json(data: Dict, filepath: Path) -> None:
    """Save dictionary as JSON."""
    import json
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data
