"""Main pipeline orchestration for power price forecasting.

This module implements Steps 2-4 of the pipeline:
- Step 2: Data Ingestion (prices + fundamentals)
- Step 3: QA Gate (checks + clean dataset)
- Step 4: Feature Engineering
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from .paths import PathBuilder, generate_run_id
from .config import load_config

logger = logging.getLogger(__name__)


class PipelineResult:
    """Container for pipeline execution results."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.success = False
        self.raw_data_path: Optional[Path] = None
        self.clean_data_path: Optional[Path] = None
        self.features_path: Optional[Path] = None
        self.qa_report_path: Optional[Path] = None
        self.qa_passed = False
        self.errors: list = []
        self.warnings: list = []
        self.stats: Dict[str, Any] = {}


def run_pipeline(
    config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    skip_qa: bool = False,
    force_refresh: bool = False,
    root_dir: Optional[Path] = None,
) -> PipelineResult:
    """
    Run the complete data pipeline (Steps 2-4).
    
    Steps:
        2. Data ingestion (prices, fundamentals)
        3. QA checks + clean dataset
        4. Feature engineering
    
    Args:
        config: Configuration dictionary
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        skip_qa: If True, skip QA gates (not recommended)
        force_refresh: If True, bypass cache and re-fetch data
        root_dir: Project root directory
    
    Returns:
        PipelineResult with paths to generated artifacts
    """
    # Initialize paths and run ID
    paths = PathBuilder(root_dir)
    paths.ensure_dirs()
    
    market = config.get("market", {}).get("code", "DE_LU")
    run_id = generate_run_id(market)
    
    result = PipelineResult(run_id)
    result.stats["start_date"] = start_date.isoformat()
    result.stats["end_date"] = end_date.isoformat()
    result.stats["market"] = market
    
    logger.info(f"="*60)
    logger.info(f"Starting pipeline run: {run_id}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Market: {market}")
    logger.info(f"="*60)
    
    try:
        # Step 2: Data Ingestion
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA INGESTION")
        logger.info("="*60)
        
        df_raw = _run_ingestion(
            config=config,
            start_date=start_date,
            end_date=end_date,
            paths=paths,
            run_id=run_id,
            force_refresh=force_refresh,
        )
        
        result.raw_data_path = paths.raw_data_dir(run_id) / "combined.parquet"
        result.stats["raw_rows"] = len(df_raw)
        result.stats["raw_columns"] = list(df_raw.columns)
        
        # Step 3: QA Gate
        logger.info("\n" + "="*60)
        logger.info("STEP 3: QA GATE")
        logger.info("="*60)
        
        if skip_qa:
            logger.warning("⚠️ QA checks SKIPPED (not recommended)")
            df_clean = df_raw.copy()
            result.qa_passed = True
        else:
            df_clean, qa_passed = _run_qa_gate(
                df=df_raw,
                config=config,
                paths=paths,
                run_id=run_id,
            )
            result.qa_passed = qa_passed
            result.qa_report_path = paths.qa_report_dir() / f"{run_id}_qa.md"
        
        result.clean_data_path = paths.clean_data_dir(run_id) / "dataset.parquet"
        result.stats["clean_rows"] = len(df_clean)
        
        # Step 4: Feature Engineering
        logger.info("\n" + "="*60)
        logger.info("STEP 4: FEATURE ENGINEERING")
        logger.info("="*60)
        
        df_features = _run_feature_engineering(
            df=df_clean,
            config=config,
            paths=paths,
            run_id=run_id,
        )
        
        result.features_path = paths.feature_data_dir(run_id) / "features.parquet"
        result.stats["feature_rows"] = len(df_features)
        result.stats["feature_columns"] = len(df_features.columns)
        
        # Success!
        result.success = True
        
        logger.info("\n" + "="*60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Raw data: {result.raw_data_path}")
        logger.info(f"Clean data: {result.clean_data_path}")
        logger.info(f"Features: {result.features_path}")
        logger.info(f"QA report: {result.qa_report_path}")
        
    except Exception as e:
        result.success = False
        result.errors.append(str(e))
        logger.error(f"❌ Pipeline failed: {e}")
        raise
    
    return result


def _run_ingestion(
    config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    paths: PathBuilder,
    run_id: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Run data ingestion step."""
    from ingest import fetch_day_ahead_prices, fetch_fundamentals
    
    # Handle nested market config structure
    market_config = config["market"].get("market", config["market"])
    market = market_config["code"]
    cache_dir = paths.cache_dir()
    
    # Fetch prices
    logger.info("Fetching day-ahead prices...")
    df_prices = fetch_day_ahead_prices(
        market=market,
        start_date=start_date,
        end_date=end_date,
        config=config,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    logger.info(f"  → Retrieved {len(df_prices)} price records")
    
    # Fetch fundamentals
    logger.info("Fetching fundamentals (load, wind, solar forecasts)...")
    df_fundamentals = fetch_fundamentals(
        market=market,
        start_date=start_date,
        end_date=end_date,
        config=config,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    logger.info(f"  → Retrieved {len(df_fundamentals)} fundamental records")
    
    # Join prices and fundamentals
    logger.info("Joining prices and fundamentals...")
    df_combined = df_prices.join(df_fundamentals, how='outer')
    df_combined = df_combined.sort_index()
    logger.info(f"  → Combined dataset: {len(df_combined)} rows")
    
    # Save raw data
    paths.ensure_run_dirs(run_id)
    raw_dir = paths.raw_data_dir(run_id)
    
    df_prices.to_parquet(raw_dir / "prices.parquet")
    df_fundamentals.to_parquet(raw_dir / "fundamentals.parquet")
    df_combined.to_parquet(raw_dir / "combined.parquet")
    
    logger.info(f"  → Saved raw data to {raw_dir}")
    
    return df_combined


def _run_qa_gate(
    df: pd.DataFrame,
    config: Dict[str, Any],
    paths: PathBuilder,
    run_id: str,
) -> Tuple[pd.DataFrame, bool]:
    """Run QA gate step."""
    from qa import run_qa_pipeline, format_qa_summary_for_log
    
    qa_report_dir = paths.qa_report_dir()
    
    # Run QA checks
    df_clean, results = run_qa_pipeline(
        df=df,
        config=config,
        run_id=run_id,
        output_dir=qa_report_dir,
        fail_fast=True,  # Fail on critical issues
    )
    
    # Log summary
    logger.info(format_qa_summary_for_log(results))
    
    # Save clean dataset
    clean_dir = paths.clean_data_dir(run_id)
    clean_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(clean_dir / "dataset.parquet")
    logger.info(f"  → Saved clean dataset to {clean_dir / 'dataset.parquet'}")
    
    qa_passed = all(r.passed for r in results)
    return df_clean, qa_passed


def _run_feature_engineering(
    df: pd.DataFrame,
    config: Dict[str, Any],
    paths: PathBuilder,
    run_id: str,
) -> pd.DataFrame:
    """Run feature engineering step."""
    from features import build_features, validate_no_leakage, get_feature_columns
    
    # Build features
    feature_dir = paths.feature_data_dir(run_id)
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "features.parquet"
    
    df_features = build_features(
        df=df,
        config=config,
        output_path=feature_path,
    )
    
    # Validate no leakage
    validate_no_leakage(df_features, config)
    
    # Log feature info
    feature_cols = get_feature_columns(df_features, config)
    logger.info(f"  → Built {len(feature_cols)} feature columns")
    logger.info(f"  → Feature matrix shape: {df_features.shape}")
    logger.info(f"  → Saved features to {feature_path}")
    
    return df_features


def run_ingestion_only(
    config: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    root_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Run only the ingestion step (for testing/debugging)."""
    paths = PathBuilder(root_dir)
    paths.ensure_dirs()
    
    market = config.get("market", {}).get("code", "DE_LU")
    run_id = generate_run_id(market)
    
    df = _run_ingestion(config, start_date, end_date, paths, run_id, force_refresh)
    return df, run_id
