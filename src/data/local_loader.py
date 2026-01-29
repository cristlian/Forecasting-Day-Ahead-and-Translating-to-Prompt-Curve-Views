"""Load features from local raw CSV files (Energy-Charts data).

This module provides functionality to load and process the raw data files
downloaded from Energy-Charts API and build features for model training.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_local_raw_data(
    raw_dir: Path,
    market: str = "DE_LU",
    resample_hourly: bool = True,
    max_gap_hours: int = 6,
) -> pd.DataFrame:
    """
    Load raw data from local CSV files.
    
    Expected files in raw_dir:
    - prices_{market}.csv: timestamp, day_ahead_price
    - load_{market}.csv: timestamp, forecast_load
    - gen_forecast_{market}.csv: timestamp, forecast_wind, forecast_solar
    
    Args:
        raw_dir: Directory containing raw CSV files
        market: Market code (e.g., 'DE_LU')
        resample_hourly: If True, resample to continuous hourly frequency
        max_gap_hours: Maximum gap size to interpolate (larger gaps stay NaN)
    
    Returns:
        Combined DataFrame with all raw data
    
    Raises:
        FileNotFoundError: If required files are missing
    """
    prices_path = raw_dir / f"prices_{market}.csv"
    load_path = raw_dir / f"load_{market}.csv"
    gen_path = raw_dir / f"gen_forecast_{market}.csv"
    
    # Check required files exist
    missing = []
    for path in [prices_path, load_path, gen_path]:
        if not path.exists():
            missing.append(path.name)
    
    if missing:
        raise FileNotFoundError(
            f"Missing required raw data files: {missing}. "
            f"Run the data download script first."
        )
    
    # Load prices
    df_prices = pd.read_csv(
        prices_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    df_prices.index = pd.to_datetime(df_prices.index, utc=True)
    logger.info(f"Loaded prices: {len(df_prices)} rows")
    
    # Load load forecast
    df_load = pd.read_csv(
        load_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    df_load.index = pd.to_datetime(df_load.index, utc=True)
    logger.info(f"Loaded load forecast: {len(df_load)} rows")
    
    # Load generation forecast (wind + solar)
    df_gen = pd.read_csv(
        gen_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    df_gen.index = pd.to_datetime(df_gen.index, utc=True)
    logger.info(f"Loaded generation forecast: {len(df_gen)} rows")
    
    # Combine all data
    df = df_prices.join(df_load, how="outer").join(df_gen, how="outer")
    df = df.sort_index()
    
    logger.info(f"Combined raw data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    
    if resample_hourly:
        # Resample to continuous hourly frequency
        df = df.resample("h").first()
        
        # Interpolate small gaps (up to max_gap_hours)
        # Use linear interpolation with a limit on consecutive NaNs
        df = df.interpolate(method="linear", limit=max_gap_hours)
        
        logger.info(f"After resampling to hourly: {len(df)} rows")
    
    return df


def build_features_from_raw(
    df_raw: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build features from raw data.
    
    Features include:
    - Calendar features (hour, day_of_week, month, etc.)
    - Lag features (24h, 48h, 168h)
    - Rolling statistics
    - Residual load
    
    Args:
        df_raw: Raw data DataFrame
        config: Optional configuration
    
    Returns:
        DataFrame with engineered features
    """
    df = df_raw.copy()
    
    # Ensure we have required columns
    required = ["day_ahead_price", "forecast_load", "forecast_wind", "forecast_solar"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Compute residual load
    df["forecast_residual_load"] = (
        df["forecast_load"] - df["forecast_wind"] - df["forecast_solar"]
    )
    
    # Calendar features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Lag features (minimum 24h to avoid look-ahead bias)
    # Using ffill with limit to handle small gaps in the lagged values
    for lag in [24, 48, 168]:  # 1 day, 2 days, 1 week
        df[f"lag_{lag}h_day_ahead_price"] = df["day_ahead_price"].shift(lag)
        df[f"lag_{lag}h_forecast_load"] = df["forecast_load"].shift(lag)
    
    # Rolling statistics (shifted by 24h to avoid look-ahead)
    # Using min_periods=1 to be more forgiving of gaps
    for window in [24, 168]:
        df[f"roll_{window}h_mean_day_ahead_price"] = (
            df["day_ahead_price"].shift(24).rolling(window, min_periods=max(1, window // 4)).mean()
        )
        df[f"roll_{window}h_std_day_ahead_price"] = (
            df["day_ahead_price"].shift(24).rolling(window, min_periods=max(1, window // 4)).std()
        )
    
    # Interaction features
    df["hour_x_dow"] = df["hour"] * df["day_of_week"]
    df["wind_x_solar"] = df["forecast_wind"] * df["forecast_solar"] / 1e6
    
    logger.info(f"Built features: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def load_local_features(
    root_dir: Optional[Path] = None,
    market: str = "DE_LU",
) -> Tuple[pd.DataFrame, str]:
    """
    Load raw data from local files and build features.
    
    This is the primary entry point for loading training data
    from the locally stored Energy-Charts CSV files.
    
    Args:
        root_dir: Project root directory
        market: Market code
    
    Returns:
        Tuple of (features DataFrame, source description)
    """
    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent
    
    raw_dir = root_dir / "data" / "raw"
    
    # Load raw data
    df_raw = load_local_raw_data(raw_dir, market)
    
    # Build features
    df_features = build_features_from_raw(df_raw)
    
    # Drop rows with NaN in target (we need the target for training)
    df_features = df_features.dropna(subset=["day_ahead_price"])
    
    # Identify the common date range where all key features are available
    # This ensures validation window has complete data
    key_features = ["forecast_load", "forecast_wind", "forecast_solar"]
    available_features = [f for f in key_features if f in df_features.columns]
    
    if available_features:
        # Find the range where all key features are non-null
        for feature in available_features:
            non_null_mask = df_features[feature].notna()
            if non_null_mask.sum() > 0:
                first_valid = df_features[non_null_mask].index.min()
                last_valid = df_features[non_null_mask].index.max()
                df_features = df_features.loc[first_valid:last_valid]
                logger.info(f"Filtered to {feature} valid range: {first_valid} to {last_valid}")
    
    logger.info(f"Loaded {len(df_features)} feature rows from local raw data")
    
    return df_features, "local_raw"
