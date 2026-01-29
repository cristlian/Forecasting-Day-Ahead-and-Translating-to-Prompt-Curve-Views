"""Model artifact management."""

import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def save_model(model, filepath: Path, metadata: Dict = None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path to save model
        metadata: Optional metadata (config, metrics, etc.)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")


def load_model(filepath: Path):
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to model file
    
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    
    return model


def load_metadata(filepath: Path) -> Dict:
    """
    Load model metadata.
    
    Args:
        filepath: Path to model file (will load .json with same name)
    
    Returns:
        Metadata dictionary
    """
    metadata_path = filepath.with_suffix('.json')
    
    if not metadata_path.exists():
        logger.warning(f"No metadata found at {metadata_path}")
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def save_feature_importance(
    importance_df: pd.DataFrame,
    filepath: Path,
    plot: bool = True
):
    """
    Save feature importance to CSV and optionally plot.
    
    Args:
        importance_df: DataFrame with feature importance
        filepath: Path to save CSV
        plot: Whether to generate a plot
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    importance_df.to_csv(filepath, index=False)
    logger.info(f"Feature importance saved to {filepath}")
    
    # Generate plot if requested
    if plot:
        import matplotlib.pyplot as plt
        
        top_n = 20
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = filepath.with_suffix('.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
