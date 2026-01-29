"""Baseline model implementation."""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class NaiveSeasonalModel:
    """
    Naive seasonal baseline model.
    
    Predicts using same-hour-last-week (168-hour lag).
    This provides a transparent, interpretable benchmark.
    """
    
    def __init__(self, lag_hours: int = 168, config: Dict = None):
        """
        Initialize baseline model.
        
        Args:
            lag_hours: Number of hours to lag (default: 168 = 1 week)
            config: Optional configuration dictionary
        """
        self.lag_hours = lag_hours
        self.name = f"Naive Seasonal (lag={lag_hours}h)"
        self.config = config or {}
        self._fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveSeasonalModel":
        """
        Fit baseline model (no-op, as it's a naive model).
        
        Args:
            X: Training features (not used)
            y: Training target
        
        Returns:
            self
        """
        logger.info(f"Fitting {self.name}")
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using lagged values.
        
        Args:
            X: Features dataframe (must include lagged target)
        
        Returns:
            Array of predictions
        """
        lag_col = f"lag_{self.lag_hours}h_day_ahead_price"
        
        if lag_col not in X.columns:
            raise ValueError(
                f"Required lag column '{lag_col}' not found in features. "
                f"Available columns: {list(X.columns)[:10]}..."
            )
        
        predictions = X[lag_col].values.copy()
        
        # Handle NaN values with forward fill from available data
        nan_mask = np.isnan(predictions)
        if nan_mask.any():
            logger.warning(f"Found {nan_mask.sum()} NaN predictions, using available lagged values")
            # Use mean of available predictions as fallback
            mean_pred = np.nanmean(predictions)
            predictions[nan_mask] = mean_pred
        
        logger.info(f"Generated {len(predictions)} predictions using {self.name}")
        
        return predictions


def train_baseline(
    df_features: pd.DataFrame,
    config: Dict,
    target_col: str = "day_ahead_price",
) -> Tuple[NaiveSeasonalModel, pd.DataFrame]:
    """
    Train baseline model and generate predictions.
    
    Args:
        df_features: Feature dataframe
        config: Configuration dictionary
        target_col: Name of target column
    
    Returns:
        Tuple of (trained model, predictions dataframe)
    """
    baseline_config = config.get("model", {}).get("baseline", {})
    lag_hours = baseline_config.get("lag_hours", 168)
    
    model = NaiveSeasonalModel(lag_hours=lag_hours, config=config)
    
    # Ensure lag column exists
    lag_col = f"lag_{lag_hours}h_{target_col}"
    if lag_col not in df_features.columns:
        logger.info(f"Creating lag column: {lag_col}")
        df_features = df_features.copy()
        df_features[lag_col] = df_features[target_col].shift(lag_hours)
    
    # Remove rows with NaN target or lag
    valid_mask = df_features[target_col].notna() & df_features[lag_col].notna()
    df_valid = df_features[valid_mask].copy()
    
    if len(df_valid) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    logger.info(f"Training on {len(df_valid)} valid samples (removed {len(df_features) - len(df_valid)} with NaN)")
    
    # Fit model (no-op for naive)
    X = df_valid.drop(columns=[target_col])
    y = df_valid[target_col]
    model.fit(X, y)
    
    # Generate predictions
    predictions = model.predict(X)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "timestamp": df_valid.index,
        "actual": y.values,
        "predicted": predictions,
        "residual": y.values - predictions,
    })
    results_df.set_index("timestamp", inplace=True)
    
    return model, results_df


def evaluate_baseline(
    df: pd.DataFrame,
    target_col: str = "day_ahead_price",
    lag_hours: int = 168
) -> pd.DataFrame:
    """
    Evaluate baseline model on historical data.
    
    Args:
        df: Dataframe with target and lagged features
        target_col: Name of target column
        lag_hours: Lag for baseline model
    
    Returns:
        DataFrame with predictions and actuals
    """
    model = NaiveSeasonalModel(lag_hours=lag_hours)
    
    # Create lag if not present
    lag_col = f"lag_{lag_hours}h_{target_col}"
    if lag_col not in df.columns:
        df = df.copy()
        df[lag_col] = df[target_col].shift(lag_hours)
    
    # Remove rows with NaN lag (first week)
    df_valid = df[df[lag_col].notna()].copy()
    
    # Predict
    predictions = model.predict(df_valid)
    
    # Create result dataframe
    results = pd.DataFrame({
        "timestamp": df_valid.index,
        "actual": df_valid[target_col].values,
        "baseline_pred": predictions,
    })
    
    return results


def save_baseline_results(
    predictions_df: pd.DataFrame,
    metrics: Dict,
    run_id: str,
    paths,
) -> Dict[str, Path]:
    """
    Save baseline model predictions and metrics.
    
    Args:
        predictions_df: DataFrame with predictions
        metrics: Dictionary of evaluation metrics
        run_id: Run identifier
        paths: PathBuilder instance
    
    Returns:
        Dictionary of saved file paths
    """
    saved_paths = {}
    
    # Save predictions
    pred_path = paths.baseline_pred(f"{run_id}.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(pred_path)
    saved_paths["predictions"] = pred_path
    logger.info(f"Saved baseline predictions to {pred_path}")
    
    # Save metrics
    metrics_path = paths.metrics_report(f"baseline_{run_id}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to metrics
    metrics_with_meta = {
        "model": "baseline_naive_seasonal",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)
    saved_paths["metrics"] = metrics_path
    logger.info(f"Saved baseline metrics to {metrics_path}")
    
    return saved_paths

