"""Stress testing and robustness analysis."""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def test_missing_driver_robustness(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    driver_columns: List[str]
) -> Dict:
    """
    Test model robustness when key drivers are missing.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        driver_columns: List of key driver columns to test
    
    Returns:
        Dictionary with results for each missing driver scenario
    """
    from .metrics import calculate_metrics
    
    logger.info("Testing missing driver robustness")
    
    # Baseline performance
    baseline_pred = model.predict(X_test)
    baseline_metrics = calculate_metrics(y_test.values, baseline_pred)
    
    results = {"baseline": baseline_metrics}
    
    # Test each driver missing
    for col in driver_columns:
        if col not in X_test.columns:
            continue
        
        X_test_missing = X_test.copy()
        X_test_missing[col] = np.nan  # Simulate missing data
        # Impute missing values with column median to avoid NaN issues
        impute_value = X_test[col].median()
        X_test_missing[col] = X_test_missing[col].fillna(impute_value)
        
        try:
            pred_missing = model.predict(X_test_missing)
            metrics_missing = calculate_metrics(y_test.values, pred_missing)
            
            # Calculate degradation
            mae_degradation_pct = (
                (metrics_missing["mae"] - baseline_metrics["mae"]) / 
                baseline_metrics["mae"] * 100
            )
            
            results[f"missing_{col}"] = {
                **metrics_missing,
                "mae_degradation_pct": mae_degradation_pct,
            }
            
            logger.info(f"Missing {col}: MAE degradation = {mae_degradation_pct:.2f}%")
        
        except Exception as e:
            logger.warning(f"Could not test missing {col}: {e}")
            results[f"missing_{col}"] = {"error": str(e)}
    
    return results


def test_volatility_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    volatility_threshold: float = 50.0
) -> Dict:
    """
    Test model performance on high vs low volatility days.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Timestamps
        volatility_threshold: Threshold for high volatility (e.g., 50 EUR/MWh std)
    
    Returns:
        Dictionary with metrics for high/low volatility periods
    """
    from .metrics import calculate_metrics
    
    logger.info("Testing volatility performance")
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    
    # Calculate daily volatility
    df["date"] = df["timestamp"].dt.date
    daily_volatility = df.groupby("date")["y_true"].std()
    
    # Classify days
    high_vol_dates = daily_volatility[daily_volatility > volatility_threshold].index
    low_vol_dates = daily_volatility[daily_volatility <= volatility_threshold].index
    
    # Calculate metrics for each group
    high_vol_mask = df["date"].isin(high_vol_dates)
    low_vol_mask = df["date"].isin(low_vol_dates)
    
    def _safe_metrics(mask):
        if mask.sum() == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "smape": float("nan"), "r2": float("nan")}
        return calculate_metrics(
            df.loc[mask, "y_true"].values,
            df.loc[mask, "y_pred"].values
        )
    
    results = {
        "high_volatility": _safe_metrics(high_vol_mask),
        "low_volatility": _safe_metrics(low_vol_mask),
        "n_high_vol_days": len(high_vol_dates),
        "n_low_vol_days": len(low_vol_dates),
    }
    
    logger.info(f"High volatility MAE: {results['high_volatility']['mae']:.2f}")
    logger.info(f"Low volatility MAE: {results['low_volatility']['mae']:.2f}")
    
    return results


def test_weekday_weekend_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> Dict:
    """
    Compare weekday vs weekend performance.
    """
    from .metrics import calculate_metrics

    df = pd.DataFrame({
        "timestamp": timestamps,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    weekday_mask = df["timestamp"].dt.dayofweek < 5
    weekend_mask = df["timestamp"].dt.dayofweek >= 5

    return {
        "weekday": calculate_metrics(df.loc[weekday_mask, "y_true"].values, df.loc[weekday_mask, "y_pred"].values),
        "weekend": calculate_metrics(df.loc[weekend_mask, "y_true"].values, df.loc[weekend_mask, "y_pred"].values),
        "n_weekday": int(weekday_mask.sum()),
        "n_weekend": int(weekend_mask.sum()),
    }


def test_extreme_price_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile_low: float = 10,
    percentile_high: float = 90
) -> Dict:
    """
    Test model performance on extreme price hours.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        percentile_low: Low percentile threshold
        percentile_high: High percentile threshold
    
    Returns:
        Dictionary with metrics for extreme price regimes
    """
    from .metrics import calculate_metrics
    
    logger.info("Testing extreme price performance")
    
    low_threshold = np.percentile(y_true, percentile_low)
    high_threshold = np.percentile(y_true, percentile_high)
    
    # Classify observations
    low_price_mask = y_true <= low_threshold
    mid_price_mask = (y_true > low_threshold) & (y_true < high_threshold)
    high_price_mask = y_true >= high_threshold
    
    results = {
        "low_prices": calculate_metrics(y_true[low_price_mask], y_pred[low_price_mask]),
        "mid_prices": calculate_metrics(y_true[mid_price_mask], y_pred[mid_price_mask]),
        "high_prices": calculate_metrics(y_true[high_price_mask], y_pred[high_price_mask]),
        "thresholds": {
            "low": float(low_threshold),
            "high": float(high_threshold),
        },
    }
    
    logger.info(f"Low price regime MAE: {results['low_prices']['mae']:.2f}")
    logger.info(f"High price regime MAE: {results['high_prices']['mae']:.2f}")
    
    return results
