"""Translate forecasts to prompt curve view and trading signals."""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def translate_to_prompt_buckets(
    forecasts: pd.DataFrame,
    bucket_definitions: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Translate hourly forecasts to prompt curve buckets.
    
    Args:
        forecasts: DataFrame with hourly forecasts (timestamp, price columns)
        bucket_definitions: Optional custom bucket definitions
    
    Returns:
        DataFrame with bucket-level fair values
    """
    logger.info("Translating forecasts to prompt buckets")
    
    if bucket_definitions is None:
        bucket_definitions = _default_bucket_definitions()
    
    # Add bucket labels
    forecasts = forecasts.copy()
    forecasts["bucket"] = forecasts.index.map(_assign_bucket)
    
    # Aggregate by bucket
    agg_dict = {"predicted_price": "mean"}
    if "actual_price" in forecasts.columns:
        agg_dict["actual_price"] = "mean"
    elif "actual" in forecasts.columns:
         # Handle case where column is named 'actual' but we want to map it
         # But usually we should rename before calling. 
         # Let's just stick to checking 'actual_price' and the user should ensure it matches
         pass

    bucket_forecasts = forecasts.groupby("bucket").agg(agg_dict).reset_index()
    
    return bucket_forecasts


def _default_bucket_definitions() -> Dict:
    """Define standard trading buckets."""
    return {
        "off_peak_night": lambda t: (t.hour >= 0) & (t.hour < 6),
        "off_peak_late": lambda t: (t.hour >= 22) & (t.hour < 24),
        "peak": lambda t: (t.hour >= 8) & (t.hour < 20) & (t.dayofweek < 5),
        "shoulder": lambda t: (
            ((t.hour >= 6) & (t.hour < 8)) |
            ((t.hour >= 20) & (t.hour < 22))
        ) & (t.dayofweek < 5),
        "weekend": lambda t: t.dayofweek >= 5,
    }


def _assign_bucket(timestamp) -> str:
    """Assign a timestamp to a trading bucket."""
    hour = timestamp.hour
    is_weekday = timestamp.dayofweek < 5
    
    if timestamp.dayofweek >= 5:
        return "weekend"
    elif hour >= 0 and hour < 6:
        return "off_peak_night"
    elif hour >= 22 and hour < 24:
        return "off_peak_late"
    elif hour >= 8 and hour < 20 and is_weekday:
        return "peak"
    else:
        return "shoulder"


def check_invalidation_rules(
    forecasts: pd.DataFrame,
    uncertainty: Optional[pd.Series] = None,
    missing_drivers: Optional[List[str]] = None,
    max_uncertainty: float = 30.0
) -> pd.Series:
    """
    Check if forecasts should be invalidated based on rules.
    
    Args:
        forecasts: DataFrame with forecasts
        uncertainty: Optional uncertainty estimates (e.g., prediction intervals)
        missing_drivers: List of missing key drivers
        max_uncertainty: Maximum acceptable uncertainty (EUR/MWh)
    
    Returns:
        Boolean series indicating which forecasts are valid
    """
    logger.info("Checking signal invalidation rules")
    
    valid = pd.Series(True, index=forecasts.index)
    
    # Rule 1: High uncertainty
    if uncertainty is not None:
        high_uncertainty = uncertainty > max_uncertainty
        valid = valid & ~high_uncertainty
        logger.info(f"Invalidated {high_uncertainty.sum()} forecasts due to high uncertainty")
    
    # Rule 2: Missing critical drivers
    if missing_drivers and len(missing_drivers) > 0:
        logger.warning(f"Missing critical drivers: {missing_drivers}")
        logger.warning("Consider invalidating all signals")
        # Could implement more sophisticated logic here
    
    # Rule 3: Extreme forecast values (sanity check)
    extreme_low = forecasts["predicted_price"] < -100  # Negative prices rare but possible
    extreme_high = forecasts["predicted_price"] > 1000  # Very high but possible
    extreme = extreme_low | extreme_high
    valid = valid & ~extreme
    logger.info(f"Invalidated {extreme.sum()} forecasts due to extreme values")
    
    logger.info(f"Valid forecasts: {valid.sum()} / {len(valid)}")
    
    return valid
