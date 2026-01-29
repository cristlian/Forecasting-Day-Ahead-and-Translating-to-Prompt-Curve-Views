"""Improved model implementation (LightGBM/XGBoost)."""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Check for LightGBM availability
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available, will fall back to sklearn")


class PowerPriceModel:
    """
    Improved power price forecasting model using gradient boosting.
    """
    
    def __init__(self, config: Dict, model_type: str = None):
        """
        Initialize model.
        
        Args:
            config: Model configuration dictionary
            model_type: 'lightgbm' or 'xgboost' (auto-selects if None)
        """
        self.config = config
        
        # Auto-select model type based on availability
        if model_type is None:
            model_type = config.get("model", {}).get("improved_model", {}).get("type", "lightgbm")
        
        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            logger.warning("LightGBM not available, falling back to sklearn GradientBoosting")
            model_type = "sklearn_gbm"
        
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self._fitted = False
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "PowerPriceModel":
        """
        Fit the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
        
        Returns:
            self
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == "lightgbm":
            self._fit_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == "xgboost":
            self._fit_xgboost(X_train, y_train, X_val, y_val)
        else:
            self._fit_sklearn_gbm(X_train, y_train, X_val, y_val)
        
        # Extract feature importance
        self._extract_feature_importance()
        
        self._fitted = True
        logger.info(f"Model trained with {len(self.feature_names)} features")
        
        return self
    
    def _fit_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ):
        """Fit LightGBM model."""
        import lightgbm as lgb
        
        model_config = self.config.get("model", {}).get("improved_model", {})
        params = model_config.get("hyperparameters", {})
        
        # Default LightGBM params if not specified
        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 8),
            "num_leaves": params.get("num_leaves", 64),
            "min_child_samples": params.get("min_child_samples", 20),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "random_state": params.get("random_state", 42),
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Setup validation data for early stopping
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append("valid")
        
        # Training callbacks
        callbacks = [lgb.log_evaluation(period=100)]
        
        early_stop_config = model_config.get("early_stopping", {})
        if early_stop_config and X_val is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stop_config.get("rounds", 50))
            )
        
        self.model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
    
    def _fit_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ):
        """Fit XGBoost model."""
        import xgboost as xgb
        
        model_config = self.config.get("model", {}).get("improved_model", {})
        params = model_config.get("hyperparameters", {})
        
        # Convert params for XGBoost naming
        xgb_params = {
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 8),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "random_state": params.get("random_state", 42),
        }
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model = xgb.XGBRegressor(**xgb_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
    
    def _fit_sklearn_gbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ):
        """Fit sklearn GradientBoostingRegressor as fallback."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        model_config = self.config.get("model", {}).get("improved_model", {})
        params = model_config.get("hyperparameters", {})
        
        self.model = GradientBoostingRegressor(
            n_estimators=min(params.get("n_estimators", 500), 200),  # Limit for speed
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=min(params.get("max_depth", 8), 6),
            subsample=params.get("subsample", 0.8),
            random_state=params.get("random_state", 42),
            verbose=0,
        )
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features dataframe
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        if self.model_type == "lightgbm":
            predictions = self.model.predict(X)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def _extract_feature_importance(self):
        """Extract and store feature importance."""
        if self.model_type == "lightgbm":
            importance = self.model.feature_importance(importance_type="gain")
            self.feature_importance = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)
        
        elif self.model_type == "xgboost":
            self.feature_importance = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)
        
        elif self.model_type == "sklearn_gbm":
            self.feature_importance = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Return only top N features
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        if top_n:
            return self.feature_importance.head(top_n)
        
        return self.feature_importance


def train_improved_model(
    df_features: pd.DataFrame,
    config: Dict,
    target_col: str = "day_ahead_price",
    train_ratio: float = 0.8,
) -> Tuple[PowerPriceModel, pd.DataFrame, pd.DataFrame]:
    """
    Train improved model with train/validation split.
    
    Uses time-based split to prevent leakage.
    
    Args:
        df_features: Feature dataframe with target
        config: Configuration dictionary
        target_col: Name of target column
        train_ratio: Fraction of data for training
    
    Returns:
        Tuple of (trained model, train predictions, val predictions)
    """
    # Remove rows with NaN
    df_clean = df_features.dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    logger.info(f"Training on {len(df_clean)} samples (removed {len(df_features) - len(df_clean)} with NaN)")
    
    # Time-based split
    split_idx = int(len(df_clean) * train_ratio)
    df_train = df_clean.iloc[:split_idx]
    df_val = df_clean.iloc[split_idx:]
    
    logger.info(f"Train: {len(df_train)} samples, Validation: {len(df_val)} samples")
    
    # Prepare features and target
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]
    
    # Train model
    model = PowerPriceModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Generate predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Create results dataframes
    train_results = pd.DataFrame({
        "timestamp": df_train.index,
        "actual": y_train.values,
        "predicted": train_preds,
        "residual": y_train.values - train_preds,
        "split": "train",
    })
    train_results.set_index("timestamp", inplace=True)
    
    val_results = pd.DataFrame({
        "timestamp": df_val.index,
        "actual": y_val.values,
        "predicted": val_preds,
        "residual": y_val.values - val_preds,
        "split": "validation",
    })
    val_results.set_index("timestamp", inplace=True)
    
    return model, train_results, val_results


def save_model_results(
    model: PowerPriceModel,
    predictions_df: pd.DataFrame,
    metrics: Dict,
    run_id: str,
    paths,
) -> Dict[str, Path]:
    """
    Save improved model, predictions, and metrics.
    
    Args:
        model: Trained model
        predictions_df: DataFrame with predictions
        metrics: Dictionary of evaluation metrics
        run_id: Run identifier
        paths: PathBuilder instance
    
    Returns:
        Dictionary of saved file paths
    """
    from .artifacts import save_model, save_feature_importance
    
    saved_paths = {}
    
    # Save model artifact
    model_path = paths.trained_model(f"model_{run_id}.bin")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path, metadata={"run_id": run_id, "model_type": model.model_type})
    saved_paths["model"] = model_path
    
    # Save predictions
    pred_path = paths.model_pred(f"{run_id}.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(pred_path)
    saved_paths["predictions"] = pred_path
    logger.info(f"Saved model predictions to {pred_path}")
    
    # Save metrics
    metrics_path = paths.metrics_report(f"model_{run_id}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_with_meta = {
        "model": f"improved_{model.model_type}",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)
    saved_paths["metrics"] = metrics_path
    logger.info(f"Saved model metrics to {metrics_path}")
    
    # Save feature importance
    if model.feature_importance is not None:
        importance_path = paths.metrics_report(f"feature_importance_{run_id}.csv")
        save_feature_importance(model.feature_importance, importance_path, plot=False)
        saved_paths["feature_importance"] = importance_path
    
    return saved_paths
