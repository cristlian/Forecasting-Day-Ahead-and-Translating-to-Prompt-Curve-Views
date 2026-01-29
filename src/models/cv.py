"""Cross-validation implementation."""

import logging
from typing import Dict, List, Tuple, Callable, Any
import pandas as pd
import numpy as np
from datetime import timedelta

logger = logging.getLogger(__name__)


class RollingOriginCV:
    """
    Rolling-origin cross-validation for time series.
    
    Maintains a fixed training window and rolls forward through time.
    This prevents leakage by ensuring test data is always after train data.
    """
    
    def __init__(
        self,
        train_size_days: int = 365,
        test_size_days: int = 7,
        gap_days: int = 0,
        n_splits: int = 10
    ):
        """
        Initialize rolling-origin CV.
        
        Args:
            train_size_days: Size of training window in days
            test_size_days: Size of test window in days
            gap_days: Gap between train and test (to avoid leakage)
            n_splits: Number of CV splits
        """
        self.train_size_days = train_size_days
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        self.n_splits = n_splits
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test splits.
        
        Args:
            df: DataFrame with datetime index
        
        Returns:
            List of (train_idx, test_idx) tuples
        """
        splits = []
        
        min_date = df.index.min()
        max_date = df.index.max()
        
        total_days = (max_date - min_date).days
        
        # Adjust train size if data is too short
        effective_train_days = min(self.train_size_days, total_days // 2)
        
        # Calculate step size to get desired number of splits
        available_days = total_days - effective_train_days - self.test_size_days
        if available_days <= 0:
            logger.warning(f"Data too short for CV. Using single split.")
            # Single split: first 80% train, last 20% test
            split_idx = int(len(df) * 0.8)
            train_idx = df.iloc[:split_idx].index
            test_idx = df.iloc[split_idx:].index
            return [(train_idx, test_idx)]
        
        step_days = max(available_days // (self.n_splits - 1), self.test_size_days) if self.n_splits > 1 else available_days
        
        for i in range(self.n_splits):
            # Calculate split dates
            train_start = min_date + timedelta(days=i * step_days)
            train_end = train_start + timedelta(days=effective_train_days)
            
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_size_days)
            
            # Check if we've exceeded available data
            if test_end > max_date:
                break
            
            # Get indices
            train_idx = df[(df.index >= train_start) & (df.index < train_end)].index
            test_idx = df[(df.index >= test_start) & (df.index < test_end)].index
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                
                logger.info(
                    f"Split {len(splits)}: Train {train_start.date()} to {train_end.date()} "
                    f"({len(train_idx)} samples) | Test {test_start.date()} to {test_end.date()} "
                    f"({len(test_idx)} samples)"
                )
        
        if not splits:
            raise ValueError("Could not create any valid CV splits from data")
        
        return splits
    
    def evaluate(
        self,
        df: pd.DataFrame,
        model_class: Callable,
        config: Dict,
        target_col: str = "day_ahead_price"
    ) -> Dict[str, Any]:
        """
        Evaluate model using rolling-origin CV.
        
        Args:
            df: Features dataframe
            model_class: Model class to instantiate (callable that takes config)
            config: Model configuration
            target_col: Name of target column
        
        Returns:
            Dictionary with CV results
        """
        try:
            from ..validation.metrics import calculate_metrics
        except ImportError:
            from validation.metrics import calculate_metrics
        
        splits = self.split(df)
        
        results = {
            "split_metrics": [],
            "predictions": [],
        }
        
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Evaluating split {split_idx + 1}/{len(splits)}")
            
            # Prepare data
            train_df = df.loc[train_idx].copy()
            test_df = df.loc[test_idx].copy()
            
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            # Remove rows with NaN (from lag/rolling features)
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]
            
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"Split {split_idx + 1} has no valid data, skipping")
                continue
            
            # Train model
            model = model_class(config)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test.values, y_pred)
            metrics["split"] = split_idx
            results["split_metrics"].append(metrics)
            
            # Store predictions
            results["predictions"].append({
                "split": split_idx,
                "timestamp": test_idx[test_valid].tolist(),
                "actual": y_test.values.tolist(),
                "predicted": y_pred.tolist(),
            })
            
            logger.info(f"Split {split_idx + 1} metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        
        if not results["split_metrics"]:
            raise ValueError("No valid CV splits could be evaluated")
        
        # Aggregate metrics
        results["mean_metrics"] = {
            metric: np.mean([s[metric] for s in results["split_metrics"]])
            for metric in results["split_metrics"][0].keys()
            if metric != "split"
        }
        
        logger.info(f"Mean CV metrics: MAE={results['mean_metrics']['mae']:.2f}, RMSE={results['mean_metrics']['rmse']:.2f}")
        
        return results
