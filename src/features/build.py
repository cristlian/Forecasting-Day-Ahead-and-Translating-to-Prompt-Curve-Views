"""Feature engineering pipeline with leakage-safe design.

CRITICAL: All features must be computable BEFORE the day-ahead auction (12:00 CET).
This means:
- Only use day-ahead FORECASTS, never actuals
- Lag features must be at least 24 hours (since we predict D+1)
- Rolling windows must only look backwards
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame, 
    config: Dict,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build all features for the model.
    
    IMPORTANT: This function is designed to be leakage-safe. All features
    are computed using only information available BEFORE the prediction target.
    
    Args:
        df: Input dataframe with clean data (UTC timestamps, has day_ahead_price + forecasts)
        config: Full configuration dictionary
        output_path: Optional path to save features
    
    Returns:
        DataFrame with engineered features aligned to target
    """
    logger.info("Building features")
    
    feature_config = config.get("features", {})
    toggles = feature_config.get("feature_toggles", {})
    
    df_features = df.copy()
    
    # Ensure index is sorted
    df_features = df_features.sort_index()
    
    # 1. Calendar features (no leakage - deterministic from timestamp)
    if toggles.get("enable_calendar", True):
        df_features = add_calendar_features(
            df_features, 
            feature_config.get("calendar_features", [])
        )
    
    # 2. Derived features (computed from forecasts)
    if toggles.get("enable_residual_load", True):
        df_features = add_derived_features(
            df_features, 
            feature_config.get("derived_features", {})
        )
    
    # 3. Lag features (CRITICAL: minimum 24h lag for D+1 prediction)
    if toggles.get("enable_lags", True):
        df_features = add_lag_features(
            df_features, 
            feature_config.get("lag_features", {})
        )
    
    # 4. Rolling features (CRITICAL: windows look backward only)
    if toggles.get("enable_rolling", True):
        df_features = add_rolling_features(
            df_features, 
            feature_config.get("rolling_features", {})
        )
    
    # 5. Interaction features
    if toggles.get("enable_interactions", True):
        df_features = add_interaction_features(
            df_features, 
            feature_config.get("interaction_features", [])
        )
    
    # Drop rows with NaN from lag/rolling features (initial warmup period)
    initial_rows = len(df_features)
    
    # Get target column
    target_col = config.get("market", {}).get("target", {}).get("column", "day_ahead_price")
    
    # Only drop where target is NaN (keep feature NaNs for now, model can handle)
    if target_col in df_features.columns:
        df_features = df_features.dropna(subset=[target_col])
    
    final_rows = len(df_features)
    logger.info(f"Feature matrix: {final_rows} rows ({initial_rows - final_rows} dropped), {len(df_features.columns)} columns")
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(output_path)
        logger.info(f"Saved features to {output_path}")
    
    return df_features


def add_calendar_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Add calendar-based features.
    
    These are deterministic from the timestamp and have no leakage.
    """
    logger.info("Adding calendar features")
    
    # Ensure we work with the index
    idx = df.index
    
    if "hour" in feature_list:
        df["hour"] = idx.hour
    
    if "day_of_week" in feature_list:
        df["day_of_week"] = idx.dayofweek
    
    if "month" in feature_list:
        df["month"] = idx.month
    
    if "is_weekend" in feature_list:
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    
    if "season" in feature_list:
        # 1=Winter (Dec,Jan,Feb), 2=Spring, 3=Summer, 4=Fall
        month = idx.month
        df["season"] = ((month % 12 + 3) // 3).astype(int)
    
    if "is_holiday" in feature_list:
        # Simple heuristic: major German holidays
        # In production, use a proper holiday calendar
        df["is_holiday"] = _get_holiday_indicator(idx)
    
    logger.debug(f"Added {len([f for f in feature_list if f in df.columns])} calendar features")
    return df


def _get_holiday_indicator(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Get holiday indicator for German market.
    
    Simple version - marks Christmas, New Year, Easter.
    In production, use 'holidays' package or proper calendar.
    """
    # Fixed holidays (approximate for Germany)
    holidays = [
        (1, 1),   # New Year
        (5, 1),   # Labor Day
        (10, 3),  # German Unity Day
        (12, 25), # Christmas
        (12, 26), # Boxing Day
    ]
    
    is_holiday = pd.Series(0, index=idx)
    
    for month, day in holidays:
        is_holiday |= ((idx.month == month) & (idx.day == day)).astype(int)
    
    return is_holiday


def add_derived_features(df: pd.DataFrame, derived_config: Dict) -> pd.DataFrame:
    """
    Add derived features computed from forecasts.
    
    These use only forecast values (not actuals) to avoid leakage.
    """
    logger.info("Adding derived features")
    
    # Forecast residual load: Load - Wind - Solar
    if "forecast_residual_load" in derived_config:
        if all(col in df.columns for col in ["forecast_load", "forecast_wind", "forecast_solar"]):
            df["forecast_residual_load"] = (
                df["forecast_load"] - df["forecast_wind"] - df["forecast_solar"]
            )
            logger.debug("Added forecast_residual_load")
    
    # Renewable share
    if "forecast_renewable_share" in derived_config:
        if all(col in df.columns for col in ["forecast_load", "forecast_wind", "forecast_solar"]):
            # Avoid division by zero
            load = df["forecast_load"].replace(0, np.nan)
            df["forecast_renewable_share"] = (
                (df["forecast_wind"] + df["forecast_solar"]) / load
            )
            logger.debug("Added forecast_renewable_share")
    
    return df


def add_lag_features(df: pd.DataFrame, lag_config: Dict) -> pd.DataFrame:
    """
    Add lagged features.
    
    CRITICAL LEAKAGE PREVENTION:
    - For D+1 prediction, minimum lag must be 24 hours
    - We're predicting prices for hours 00:00-23:00 of day D+1
    - At cutoff (12:00 CET on day D), we only know prices through the end of D-1
    - Therefore, the most recent price we can use is from hour 23 of D-1 (lag=24+)
    
    The config specifies lags: [24, 48, 168] (1 day, 2 days, 1 week)
    """
    logger.info("Adding lag features")
    
    columns = lag_config.get("columns", [])
    lags = lag_config.get("lags", [24, 48, 168])
    
    # Validate minimum lag
    min_lag = min(lags) if lags else 24
    if min_lag < 24:
        logger.warning(f"Lag of {min_lag}h may cause leakage for D+1 prediction! Minimum recommended: 24h")
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for lag features, skipping")
            continue
        
        for lag in lags:
            feature_name = f"lag_{lag}h_{col}"
            df[feature_name] = df[col].shift(lag)
            logger.debug(f"Added {feature_name}")
    
    return df


def add_rolling_features(df: pd.DataFrame, rolling_config: Dict) -> pd.DataFrame:
    """
    Add rolling window features.
    
    CRITICAL LEAKAGE PREVENTION:
    - Windows look backward only (no centered windows)
    - The window must end BEFORE the prediction target
    - For D+1 prediction with cutoff at 12:00 CET on D, we shift the window
    
    We use shift(24) to ensure the rolling window ends at hour 23 of D-1,
    which is the last hour we know when predicting D+1.
    """
    logger.info("Adding rolling features")
    
    columns = rolling_config.get("columns", [])
    windows = rolling_config.get("windows", [24, 168])
    aggregations = rolling_config.get("aggregations", ["mean", "std"])
    min_periods = rolling_config.get("min_periods", 12)
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for rolling features, skipping")
            continue
        
        for window in windows:
            for agg in aggregations:
                feature_name = f"roll_{window}h_{agg}_{col}"
                
                # Compute rolling stat, then shift to avoid leakage
                # The shift ensures we only use data available at prediction time
                rolling = df[col].rolling(window=window, min_periods=min_periods)
                
                if agg == "mean":
                    stat = rolling.mean()
                elif agg == "std":
                    stat = rolling.std()
                elif agg == "min":
                    stat = rolling.min()
                elif agg == "max":
                    stat = rolling.max()
                else:
                    logger.warning(f"Unknown aggregation: {agg}")
                    continue
                
                # Shift by 24 hours to ensure no leakage for D+1 prediction
                # This means the rolling window ends at the same hour of D-1
                df[feature_name] = stat.shift(24)
                logger.debug(f"Added {feature_name}")
    
    return df


def add_interaction_features(df: pd.DataFrame, interaction_list: List) -> pd.DataFrame:
    """
    Add interaction features.
    
    These are products of existing features - no leakage if base features are clean.
    """
    logger.info("Adding interaction features")
    
    for interaction in interaction_list:
        if len(interaction) != 2:
            logger.warning(f"Invalid interaction spec: {interaction}")
            continue
        
        col1, col2 = interaction
        
        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(f"Missing columns for interaction {col1} x {col2}")
            continue
        
        feature_name = f"interact_{col1}_{col2}"
        df[feature_name] = df[col1] * df[col2]
        logger.debug(f"Added {feature_name}")
    
    return df


def get_feature_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    """
    Get list of feature columns (excluding target and timestamp).
    
    Args:
        df: Feature DataFrame
        config: Configuration dictionary
        
    Returns:
        List of feature column names
    """
    target_col = config.get("market", {}).get("target", {}).get("column", "day_ahead_price")
    
    # Exclude target and any raw data columns
    exclude_cols = {target_col, "timestamp"}
    
    # Also exclude raw forecast columns (we use their lagged/rolled versions)
    raw_cols = {"forecast_load", "forecast_wind", "forecast_solar", "forecast_residual_load"}
    exclude_cols.update(raw_cols)
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return feature_cols


def validate_no_leakage(df: pd.DataFrame, config: Dict) -> bool:
    """
    Validate that features don't have look-ahead leakage.
    
    Simple check: ensure lag features have appropriate shifts.
    """
    logger.info("Validating feature leakage")
    
    lag_config = config.get("features", {}).get("lag_features", {})
    lags = lag_config.get("lags", [])
    
    # Check minimum lag
    if lags and min(lags) < 24:
        logger.error(f"Lag features use lags < 24h, which may cause leakage!")
        return False
    
    # Check that we have the expected lag columns
    for lag in lags:
        lag_cols = [c for c in df.columns if f"lag_{lag}h_" in c]
        if not lag_cols:
            logger.warning(f"No lag-{lag}h features found")
    
    logger.info("Leakage validation passed")
    return True
