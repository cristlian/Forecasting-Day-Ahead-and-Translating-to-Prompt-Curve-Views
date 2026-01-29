"""Training orchestration for baseline and improved models.

This module handles:
- Loading features from cache or sample data
- Training baseline and improved models
- Generating predictions and metrics
- Saving artifacts
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingResult:
    """Container for training results."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.success = False
        self.model_type: str = ""
        self.predictions_path: Optional[Path] = None
        self.metrics_path: Optional[Path] = None
        self.model_path: Optional[Path] = None
        self.feature_importance_path: Optional[Path] = None
        self.metrics: Dict[str, float] = {}
        self.errors: list = []


class CacheMissingError(Exception):
    """Raised when cache-only mode is used but cache is not available."""
    pass


def _get_paths_module():
    """Import paths module handling both relative and absolute imports."""
    try:
        from .paths import PathBuilder, generate_run_id
        return PathBuilder, generate_run_id
    except ImportError:
        from pipeline.paths import PathBuilder, generate_run_id
        return PathBuilder, generate_run_id


def _get_sample_module():
    """Import sample data module handling both relative and absolute imports."""
    try:
        from ..data.sample import load_sample_features
        return load_sample_features
    except ImportError:
        from data.sample import load_sample_features
        return load_sample_features


def _get_local_loader():
    """Import local loader module."""
    try:
        from ..data.local_loader import load_local_features
        return load_local_features
    except ImportError:
        from data.local_loader import load_local_features
        return load_local_features


def _get_models_modules():
    """Import model modules handling both relative and absolute imports."""
    try:
        from ..models.baseline import train_baseline, save_baseline_results
        from ..models.model import train_improved_model, save_model_results
        from ..validation.metrics import calculate_metrics
    except ImportError:
        from models.baseline import train_baseline, save_baseline_results
        from models.model import train_improved_model, save_model_results
        from validation.metrics import calculate_metrics
    return train_baseline, save_baseline_results, train_improved_model, save_model_results, calculate_metrics


def find_features_cache(
    paths,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Find cached features file.
    
    Args:
        paths: PathBuilder instance
        run_id: Specific run ID to look for (optional)
    
    Returns:
        Path to features file if found, None otherwise
    """
    features_dir = paths.data / "features"
    
    if run_id:
        # Look for specific run
        run_dir = features_dir / run_id
        features_path = run_dir / "features.parquet"
        if features_path.exists():
            return features_path
    
    # Look for any recent features
    if features_dir.exists():
        # Find all run directories with features
        feature_files = list(features_dir.glob("*/features.parquet"))
        if feature_files:
            # Return most recent
            return max(feature_files, key=lambda p: p.stat().st_mtime)
    
    return None


def load_features(
    paths,
    cache_only: bool = False,
    use_sample: bool = False,
    run_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Load feature data for training.
    
    Priority:
    1. If use_sample: generate/load sample data (for testing only)
    2. Try local raw data files (data/raw/*.csv)
    3. Try cached features (data/features/)
    4. If cache_only and nothing found: error
    
    Args:
        paths: PathBuilder instance
        cache_only: Only use cached data
        use_sample: Use synthetic sample data (testing only)
        run_id: Specific run ID to load
    
    Returns:
        Tuple of (features DataFrame, source description)
    
    Raises:
        CacheMissingError: If no data available
    """
    if use_sample:
        load_sample_features = _get_sample_module()
        logger.info("Loading sample features (synthetic data for offline testing)")
        df = load_sample_features(paths.root)
        return df, "sample"
    
    # Try local raw data first (primary source)
    try:
        load_local_features = _get_local_loader()
        df, source = load_local_features(paths.root)
        logger.info(f"Loaded {len(df)} rows from local raw data")
        return df, source
    except FileNotFoundError as e:
        logger.debug(f"Local raw data not found: {e}")
    except Exception as e:
        logger.warning(f"Error loading local raw data: {e}")
    
    # Try to find cached features
    cache_path = find_features_cache(paths, run_id)
    
    if cache_path is not None:
        logger.info(f"Loading features from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        return df, f"cache:{cache_path}"
    
    if cache_only:
        raise CacheMissingError(
            "No cached features found. Options:\n"
            "  1. Ensure raw data files exist in data/raw/\n"
            "  2. Run full pipeline first: python -m pipeline run --start-date ... --end-date ...\n"
            "  3. Use sample data: python -m pipeline train --use-sample"
        )
    
    raise CacheMissingError(
        "No features available. Ensure raw data exists in data/raw/ or run:\n"
        "  python -m pipeline run --start-date 2024-01-01 --end-date 2024-12-31\n"
        "  OR for testing:\n"
        "  python -m pipeline train --use-sample"
    )


def run_training(
    config: Dict[str, Any],
    model_type: str = "both",
    cache_only: bool = False,
    use_sample: bool = False,
    root_dir: Optional[Path] = None,
) -> Dict[str, TrainingResult]:
    """
    Run model training.
    
    Args:
        config: Configuration dictionary
        model_type: 'baseline', 'improved', or 'both'
        cache_only: Only use cached data (no API calls)
        use_sample: Use synthetic sample data
        root_dir: Project root directory
    
    Returns:
        Dictionary of model_type -> TrainingResult
    """
    PathBuilder, generate_run_id = _get_paths_module()
    train_baseline, save_baseline_results, train_improved_model, save_model_results, calculate_metrics = _get_models_modules()
    
    # Initialize paths
    paths = PathBuilder(root_dir)
    paths.ensure_dirs()
    
    market = config.get("market", {}).get("market", {}).get("code", "DE_LU")
    run_id = generate_run_id(market)
    
    logger.info(f"="*60)
    logger.info(f"Starting training run: {run_id}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Cache only: {cache_only}")
    logger.info(f"Use sample: {use_sample}")
    logger.info(f"="*60)
    
    results = {}
    
    # Load features
    try:
        df_features, source = load_features(
            paths=paths,
            cache_only=cache_only,
            use_sample=use_sample,
        )
        logger.info(f"Loaded {len(df_features)} samples from {source}")
    except CacheMissingError as e:
        logger.error(str(e))
        raise
    
    target_col = config.get("market", {}).get("target", {}).get("column", "day_ahead_price")
    
    # Train baseline
    if model_type in ["baseline", "both"]:
        logger.info("\n" + "="*60)
        logger.info("TRAINING BASELINE MODEL")
        logger.info("="*60)
        
        result = TrainingResult(run_id)
        result.model_type = "baseline"
        
        try:
            model, predictions_df = train_baseline(
                df_features=df_features,
                config=config,
                target_col=target_col,
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                predictions_df["actual"].values,
                predictions_df["predicted"].values,
            )
            result.metrics = metrics
            
            logger.info(f"Baseline metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
            
            # Save results
            saved = save_baseline_results(
                predictions_df=predictions_df,
                metrics=metrics,
                run_id=run_id,
                paths=paths,
            )
            
            result.predictions_path = saved["predictions"]
            result.metrics_path = saved["metrics"]
            result.success = True
            
        except Exception as e:
            logger.error(f"Baseline training failed: {e}")
            result.errors.append(str(e))
        
        results["baseline"] = result
    
    # Train improved model
    if model_type in ["improved", "both"]:
        logger.info("\n" + "="*60)
        logger.info("TRAINING IMPROVED MODEL")
        logger.info("="*60)
        
        result = TrainingResult(run_id)
        result.model_type = "improved"
        
        try:
            model, train_preds, val_preds = train_improved_model(
                df_features=df_features,
                config=config,
                target_col=target_col,
            )
            
            # Combine predictions
            all_preds = pd.concat([train_preds, val_preds])
            
            # Calculate validation metrics (more representative)
            metrics = calculate_metrics(
                val_preds["actual"].values,
                val_preds["predicted"].values,
            )
            result.metrics = metrics
            
            logger.info(f"Improved model metrics (val): MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
            
            # Save results
            saved = save_model_results(
                model=model,
                predictions_df=all_preds,
                metrics=metrics,
                run_id=run_id,
                paths=paths,
            )
            
            result.predictions_path = saved["predictions"]
            result.metrics_path = saved["metrics"]
            result.model_path = saved.get("model")
            result.feature_importance_path = saved.get("feature_importance")
            result.success = True
            
        except Exception as e:
            logger.error(f"Improved model training failed: {e}")
            result.errors.append(str(e))
            import traceback
            traceback.print_exc()
        
        results["improved"] = result
    
    # Generate validation report
    if len(results) > 0:
        try:
            generate_validation_report(results, run_id, paths, config)
        except Exception as e:
            logger.warning(f"Failed to generate validation report: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if result.success:
            logger.info(f"  Predictions: {result.predictions_path}")
            logger.info(f"  Metrics: {result.metrics_path}")
    
    return results


def generate_validation_report(
    results: Dict[str, TrainingResult],
    run_id: str,
    paths,
    config: Dict,
) -> Path:
    """
    Generate validation comparison report.
    
    Args:
        results: Dictionary of training results
        run_id: Run identifier
        paths: PathBuilder instance
        config: Configuration dictionary
    
    Returns:
        Path to generated report
    """
    report_path = paths.validation_report(f"{run_id}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# Model Validation Report",
        f"",
        f"**Run ID:** {run_id}",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        f"",
        f"## Metrics Comparison",
        f"",
        f"| Model | MAE | RMSE | sMAPE | R² |",
        f"|-------|-----|------|-------|-----|",
    ]
    
    for name, result in results.items():
        if result.success and result.metrics:
            m = result.metrics
            lines.append(
                f"| {name} | {m.get('mae', 'N/A'):.2f} | {m.get('rmse', 'N/A'):.2f} | "
                f"{m.get('smape', 'N/A'):.2f}% | {m.get('r2', 'N/A'):.3f} |"
            )
    
    # Add leakage check description
    lines.extend([
        f"",
        f"## Leakage Prevention",
        f"",
        f"This pipeline implements several safeguards against data leakage:",
        f"",
        f"1. **Time-based split**: Training/validation split respects temporal ordering",
        f"2. **Lag features**: Minimum 24-hour lag ensures features are available before D+1 auction",
        f"3. **Rolling windows**: All rolling calculations look backward only (shift before roll)",
        f"4. **Forecast-only fundamentals**: Uses day-ahead forecasts, not actual values",
        f"",
        f"## Output Artifacts",
        f"",
    ])
    
    for name, result in results.items():
        lines.append(f"### {name.title()} Model")
        if result.predictions_path:
            lines.append(f"- Predictions: `{result.predictions_path}`")
        if result.metrics_path:
            lines.append(f"- Metrics: `{result.metrics_path}`")
        if result.model_path:
            lines.append(f"- Model artifact: `{result.model_path}`")
        if result.feature_importance_path:
            lines.append(f"- Feature importance: `{result.feature_importance_path}`")
        lines.append("")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Validation report saved to {report_path}")
    
    return report_path
