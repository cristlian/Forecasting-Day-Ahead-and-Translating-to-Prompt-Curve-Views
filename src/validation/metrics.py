"""Evaluation metrics for model validation."""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    
    return metrics


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        sMAPE value (0-100%)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    
    # Avoid division by zero
    mask = denominator != 0
    smape_values = np.zeros_like(y_true, dtype=float)
    smape_values[mask] = (diff[mask] / denominator[mask]) * 100
    
    return np.mean(smape_values)


def calculate_bucket_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    buckets: Dict[str, callable]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different time buckets (peak, off-peak, etc.).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Timestamps for each prediction
        buckets: Dictionary of bucket_name -> filter_function
    
    Returns:
        Dictionary of bucket_name -> metrics
    """
    import pandas as pd
    
    # Ensure timestamps is a DatetimeIndex for consistent access
    if isinstance(timestamps, pd.DatetimeIndex):
        ts_index = timestamps
    else:
        ts_index = pd.to_datetime(timestamps, utc=True)
    
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    }, index=ts_index)
    
    bucket_metrics = {}
    
    for bucket_name, filter_func in buckets.items():
        try:
            # Try calling filter with the index directly
            mask = filter_func(df.index)
        except (AttributeError, TypeError):
            # Fallback: try with Series wrapper
            try:
                mask = filter_func(df.index.to_series())
            except Exception:
                continue
        
        if not isinstance(mask, (pd.Series, np.ndarray)):
            continue
            
        n_samples = mask.sum() if hasattr(mask, 'sum') else sum(mask)
        
        if n_samples == 0:
            continue
        
        bucket_metrics[bucket_name] = calculate_metrics(
            df.loc[mask, "y_true"].values,
            df.loc[mask, "y_pred"].values
        )
        bucket_metrics[bucket_name]["n_samples"] = int(n_samples)
    
    return bucket_metrics


def define_trading_buckets():
    """
    Define standard trading time buckets.
    
    Returns:
        Dictionary of bucket definitions
    """
    import pandas as pd
    
    buckets = {
        "off_peak_night": lambda t: (t.hour >= 0) & (t.hour < 6),
        "off_peak_late": lambda t: (t.hour >= 22) & (t.hour < 24),
        "peak": lambda t: (t.hour >= 8) & (t.hour < 20) & (t.dayofweek < 5),
        "shoulder": lambda t: ~((t.hour >= 0) & (t.hour < 6)) & 
                               ~((t.hour >= 22) & (t.hour < 24)) &
                               ~((t.hour >= 8) & (t.hour < 20) & (t.dayofweek < 5)),
        "weekend": lambda t: t.dayofweek >= 5,
        "weekday": lambda t: t.dayofweek < 5,
    }
    
    return buckets


def define_hour_buckets() -> Dict[str, callable]:
    """
    Define 24 hourly buckets for hour-of-day metrics.

    Returns:
        Dictionary of hour bucket definitions
    """
    buckets = {}
    for hour in range(24):
        # Handle both pd.Series and pd.DatetimeIndex 
        buckets[f"hour_{hour:02d}"] = lambda t, h=hour: (
            t.dt.hour == h if hasattr(t, 'dt') else t.hour == h
        )
    return buckets


def calculate_baseline_improvement(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate improvement of model over baseline.
    
    Args:
        baseline_metrics: Metrics from baseline model
        model_metrics: Metrics from improved model
    
    Returns:
        Dictionary of improvement percentages
    """
    improvement = {}
    
    for metric in ["mae", "rmse", "smape"]:
        if metric in baseline_metrics and metric in model_metrics:
            baseline_val = baseline_metrics[metric]
            model_val = model_metrics[metric]
            
            # Lower is better for these metrics
            improvement[f"{metric}_improvement_pct"] = (
                (baseline_val - model_val) / baseline_val * 100
            )
    
    return improvement
