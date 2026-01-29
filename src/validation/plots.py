"""Validation plotting functions."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    output_path: Path,
    title: str = "Predictions vs Actual"
):
    """
    Plot time series of predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Timestamps
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(timestamps, y_true, label="Actual", alpha=0.7, linewidth=1)
    ax.plot(timestamps, y_pred, label="Predicted", alpha=0.7, linewidth=1)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved predictions plot to {output_path}")


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Predicted vs Actual"
):
    """
    Plot scatter of predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect prediction")
    
    ax.set_xlabel("Actual Price (EUR/MWh)")
    ax.set_ylabel("Predicted Price (EUR/MWh)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved scatter plot to {output_path}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    output_path: Path
):
    """
    Plot residuals over time and distribution.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Timestamps
        output_path: Path to save figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series of residuals
    axes[0].plot(timestamps, residuals, alpha=0.5, linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual (EUR/MWh)")
    axes[0].set_title("Residuals Over Time")
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Residual (EUR/MWh)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved residuals plot to {output_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save figure
        top_n: Number of top features to plot
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved feature importance plot to {output_path}")


def plot_bucket_metrics(
    bucket_metrics: dict,
    metric: str,
    output_path: Path
):
    """
    Plot metrics across different time buckets.
    
    Args:
        bucket_metrics: Dictionary of bucket -> metrics
        metric: Which metric to plot (e.g., 'mae', 'rmse')
        output_path: Path to save figure
    """
    buckets = list(bucket_metrics.keys())
    values = [bucket_metrics[b][metric] for b in buckets]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(len(buckets)), values)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} by Time Bucket')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved bucket metrics plot to {output_path}")
