"""Feature engineering pipeline."""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Build all features for the model.
    
    Args:
        df: Input dataframe with raw data
        config: Feature configuration
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("Building features")
    
    feature_config = config["features"]
    df_features = df.copy()
    
    # Calendar features
    if feature_config["feature_toggles"]["enable_calendar"]:
        df_features = add_calendar_features(df_features, feature_config["calendar_features"])
    
    # Lag features
    if feature_config["feature_toggles"]["enable_lags"]:
        df_features = add_lag_features(df_features, feature_config["lag_features"])
    
    # Rolling features
    if feature_config["feature_toggles"]["enable_rolling"]:
        df_features = add_rolling_features(df_features, feature_config["rolling_features"])
    
    # Derived features
    if feature_config["feature_toggles"]["enable_residual_load"]:
        df_features = add_derived_features(df_features, feature_config["derived_features"])
    
    # Interaction features
    if feature_config["feature_toggles"]["enable_interactions"]:
        df_features = add_interaction_features(df_features, feature_config["interaction_features"])
    
    logger.info(f"Built {len(df_features.columns)} total features")
    
    return df_features


def add_calendar_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Add calendar-based features."""
    logger.info("Adding calendar features")
    
    if "hour" in feature_list:
        df["hour"] = df.index.hour
    
    if "day_of_week" in feature_list:
        df["day_of_week"] = df.index.dayofweek
    
    if "month" in feature_list:
        df["month"] = df.index.month
    
    if "is_weekend" in feature_list:
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    if "season" in feature_list:
        month = df.index.month
        df["season"] = ((month % 12 + 3) // 3).astype(int)  # 1=Winter, 2=Spring, etc.
    
    # TODO: Add holiday calendar
    if "is_holiday" in feature_list:
        df["is_holiday"] = 0  # Placeholder
    
    return df


def add_lag_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add lagged features."""
    logger.info("Adding lag features")
    
    for col in config["columns"]:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for lag features")
            continue
        
        for lag in config["lags"]:
            df[f"lag_{lag}h_{col}"] = df[col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add rolling window features."""
    logger.info("Adding rolling features")
    
    for col in config["columns"]:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for rolling features")
            continue
        
        for window in config["windows"]:
            for agg in config["aggregations"]:
                feature_name = f"rolling_{window}h_{agg}_{col}"
                
                if agg == "mean":
                    df[feature_name] = df[col].rolling(window=window).mean()
                elif agg == "std":
                    df[feature_name] = df[col].rolling(window=window).std()
                elif agg == "min":
                    df[feature_name] = df[col].rolling(window=window).min()
                elif agg == "max":
                    df[feature_name] = df[col].rolling(window=window).max()
    
    return df


def add_derived_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add derived features based on formulas."""
    logger.info("Adding derived features")
    
    # Residual load
    if "residual_load" in config:
        required = ["actual_load", "wind_generation", "solar_generation"]
        if all(col in df.columns for col in required):
            df["residual_load"] = (
                df["actual_load"] - df["wind_generation"] - df["solar_generation"]
            )
        else:
            logger.warning("Cannot compute residual_load: missing required columns")
    
    # Renewable share
    if "renewable_share" in config:
        if all(col in df.columns for col in ["actual_load", "wind_generation", "solar_generation"]):
            df["renewable_share"] = (
                (df["wind_generation"] + df["solar_generation"]) / df["actual_load"]
            ).clip(0, 1)
        else:
            logger.warning("Cannot compute renewable_share: missing required columns")
    
    return df


def add_interaction_features(df: pd.DataFrame, interactions: List) -> pd.DataFrame:
    """Add interaction features."""
    logger.info("Adding interaction features")
    
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        else:
            logger.warning(f"Cannot create interaction {col1}_x_{col2}: columns missing")
    
    return df
