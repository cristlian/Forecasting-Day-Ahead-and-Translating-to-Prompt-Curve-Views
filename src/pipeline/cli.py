"""Command-line interface for the power fair value pipeline."""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

from .run import run_pipeline, PipelineResult
from .config import load_config


def setup_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Power Fair Value Forecasting Pipeline - DE-LU Market",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (ingestion -> QA -> features) - requires API key or cache
    python -m pipeline run --start-date 2024-01-01 --end-date 2024-01-31

    # Train models using cached features (no API key needed)
    python -m pipeline train --cache-only

    # Train models using sample data (no API key needed, works on fresh clone)
    python -m pipeline train --use-sample

    # Train only baseline model
    python -m pipeline train --model baseline --use-sample

    # Evaluate models
    python -m pipeline eval --use-sample

    # Validate with stress tests
    python -m pipeline validate --date 2026-01-29 --use-sample

Environment Variables (OPTIONAL):
  ENTSOE_API_KEY    API key for ENTSO-E Transparency Platform
                    Get your key at: https://transparency.entsoe.eu/
                    NOT required for --cache-only or --use-sample modes
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run the data pipeline (Steps 2-4)")
    run_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for data retrieval (YYYY-MM-DD)"
    )
    run_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for data retrieval (YYYY-MM-DD)"
    )
    run_parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to configuration directory (default: config)"
    )
    run_parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA gate (not recommended for production)"
    )
    run_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache and re-fetch all data"
    )
    run_parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion step (use existing cached data)"
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models (Steps 5-6)")
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "improved", "both"],
        default="both",
        help="Which model to train (default: both)"
    )
    train_parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached features (fail if not available)"
    )
    train_parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use synthetic sample data (works without API keys)"
    )
    train_parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to configuration directory"
    )
    train_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    # Eval command  
    eval_parser = subparsers.add_parser("eval", help="Evaluate models with cross-validation")
    eval_parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "improved", "both"],
        default="both",
        help="Which model to evaluate (default: both)"
    )
    eval_parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached features"
    )
    eval_parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use synthetic sample data"
    )
    eval_parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to configuration directory"
    )
    eval_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    # Validate command (Step 7)
    validate_parser = subparsers.add_parser("validate", help="Run validation + stress tests (Step 7)")
    validate_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Validation date (YYYY-MM-DD) used for deterministic run_id"
    )
    validate_parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached features (fail if not available)"
    )
    validate_parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use synthetic sample data (works without API keys)"
    )
    validate_parser.add_argument(
        "--llm-test",
        action="store_true",
        help="Run optional LLM test (skips if API key missing)"
    )
    validate_parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to configuration directory"
    )
    validate_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(verbose=getattr(args, 'verbose', False))
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == "run":
            return _handle_run(args, logger)
        
        elif args.command == "train":
            return _handle_train(args, logger)
        
        elif args.command == "eval":
            return _handle_eval(args, logger)
        
        elif args.command == "validate":
            return _handle_validate(args, logger)
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1


def _handle_run(args, logger) -> int:
    """Handle the 'run' command."""
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.error("Use YYYY-MM-DD format (e.g., 2024-01-15)")
        return 1
    
    if start_date > end_date:
        logger.error("Start date must be before end date")
        return 1
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_dir}")
    config = load_config(args.config_dir)
    
    market = config.get("market", {}).get("market", {}).get("code", "unknown")
    logger.info(f"Market: {market}")
    
    # Check for API key
    import os
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key and not args.skip_ingest:
        logger.warning(
            "⚠️  ENTSOE_API_KEY not set. Will attempt to use cached data or SMARD fallback."
        )
        logger.warning(
            "   Get your API key at: https://transparency.entsoe.eu/"
        )
        logger.warning(
            "   Or use --skip-ingest if you have cached data."
        )
    
    # Run pipeline
    result = run_pipeline(
        config=config,
        start_date=start_date,
        end_date=end_date,
        skip_qa=args.skip_qa,
        force_refresh=args.force_refresh,
    )
    
    # Print summary
    _print_result_summary(result, logger)
    
    return 0 if result.success else 1


def _handle_train(args, logger) -> int:
    """Handle the 'train' command."""
    from .train import run_training, CacheMissingError
    
    logger.info(f"Loading configuration from {args.config_dir}")
    config = load_config(args.config_dir)
    
    try:
        results = run_training(
            config=config,
            model_type=args.model,
            cache_only=args.cache_only,
            use_sample=args.use_sample,
        )
        
        # Check if all succeeded
        all_success = all(r.success for r in results.values())
        return 0 if all_success else 1
        
    except CacheMissingError as e:
        logger.error(f"\n{e}")
        return 1


def _handle_eval(args, logger) -> int:
    """Handle the 'eval' command with cross-validation."""
    from .train import load_features, CacheMissingError
    from .paths import PathBuilder, generate_run_id
    try:
        from ..models.cv import RollingOriginCV
        from ..models.baseline import NaiveSeasonalModel
        from ..models.model import PowerPriceModel
        from ..validation.metrics import calculate_metrics
    except ImportError:
        from models.cv import RollingOriginCV
        from models.baseline import NaiveSeasonalModel
        from models.model import PowerPriceModel
        from validation.metrics import calculate_metrics
    import json
    
    logger.info(f"Loading configuration from {args.config_dir}")
    config = load_config(args.config_dir)
    
    paths = PathBuilder()
    paths.ensure_dirs()
    
    try:
        df_features, source = load_features(
            paths=paths,
            cache_only=args.cache_only,
            use_sample=args.use_sample,
        )
        logger.info(f"Loaded {len(df_features)} samples from {source}")
    except CacheMissingError as e:
        logger.error(f"\n{e}")
        return 1
    
    target_col = config.get("market", {}).get("target", {}).get("column", "day_ahead_price")
    cv_config = config.get("model", {}).get("cross_validation", {})
    
    # Setup CV
    cv = RollingOriginCV(
        train_size_days=cv_config.get("train_size_days", 30),
        test_size_days=cv_config.get("test_size_days", 7),
        gap_days=cv_config.get("gap_days", 0),
        n_splits=min(cv_config.get("n_splits", 5), 3),  # Limit splits for sample data
    )
    
    market = config.get("market", {}).get("market", {}).get("code", "DE_LU")
    run_id = generate_run_id(market)
    
    results = {}
    
    if args.model in ["baseline", "both"]:
        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION: BASELINE MODEL")
        logger.info("="*60)
        
        cv_results = cv.evaluate(
            df=df_features,
            model_class=lambda cfg: NaiveSeasonalModel(config=cfg),
            config=config,
            target_col=target_col,
        )
        results["baseline"] = cv_results
        logger.info(f"Baseline CV metrics: {cv_results['mean_metrics']}")
    
    if args.model in ["improved", "both"]:
        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION: IMPROVED MODEL")
        logger.info("="*60)
        
        cv_results = cv.evaluate(
            df=df_features,
            model_class=PowerPriceModel,
            config=config,
            target_col=target_col,
        )
        results["improved"] = cv_results
        logger.info(f"Improved model CV metrics: {cv_results['mean_metrics']}")
    
    # Save CV results
    cv_results_path = paths.metrics_report(f"cv_results_{run_id}.json")
    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable = {}
    for model_name, model_results in results.items():
        serializable[model_name] = {
            "mean_metrics": model_results["mean_metrics"],
            "split_metrics": model_results["split_metrics"],
        }
    
    with open(cv_results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    logger.info(f"\nCV results saved to {cv_results_path}")
    
    return 0


def _handle_validate(args, logger) -> int:
    """Handle the 'validate' command (Step 7)."""
    try:
        from ..validation.runner import run_validation
    except ImportError:
        from validation.runner import run_validation
    from .train import CacheMissingError

    logger.info(f"Loading configuration from {args.config_dir}")

    try:
        config = load_config(args.config_dir)
    except Exception as e:
        logger.error(f"✗ Configuration invalid: {e}")
        return 1

    try:
        result = run_validation(
            config=config,
            validation_date=args.date,
            cache_only=args.cache_only,
            use_sample=args.use_sample,
            llm_test=args.llm_test,
        )
        logger.info(f"Validation report: {result.report_path}")
        logger.info(f"Validation metrics: {result.metrics_path}")
        return 0
    except CacheMissingError as e:
        logger.error(f"\n{e}")
        return 1
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def _print_result_summary(result: PipelineResult, logger):
    """Print a summary of pipeline results."""
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Run ID: {result.run_id}")
    logger.info(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    logger.info(f"QA Gate: {'✅ PASSED' if result.qa_passed else '❌ FAILED'}")
    
    if result.stats:
        logger.info("\nStatistics:")
        for key, value in result.stats.items():
            if isinstance(value, list):
                logger.info(f"  {key}: {len(value)} items")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("\nOutput Artifacts:")
    if result.raw_data_path:
        logger.info(f"  Raw data: {result.raw_data_path}")
    if result.clean_data_path:
        logger.info(f"  Clean data: {result.clean_data_path}")
    if result.features_path:
        logger.info(f"  Features: {result.features_path}")
    if result.qa_report_path:
        logger.info(f"  QA report: {result.qa_report_path}")
    
    if result.errors:
        logger.error("\nErrors:")
        for error in result.errors:
            logger.error(f"  - {error}")
    
    if result.warnings:
        logger.warning("\nWarnings:")
        for warning in result.warnings:
            logger.warning(f"  - {warning}")


if __name__ == "__main__":
    sys.exit(main())
