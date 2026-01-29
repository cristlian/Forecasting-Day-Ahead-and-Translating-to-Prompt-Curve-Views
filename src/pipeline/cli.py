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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Power Fair Value Forecasting Pipeline - DE-LU Market",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline for specific date range
  python -m src.pipeline.cli run --start-date 2024-01-01 --end-date 2024-01-31

  # Run with verbose logging
  python -m src.pipeline.cli run --start-date 2024-01-01 --end-date 2024-01-31 -v

  # Force refresh (bypass cache)
  python -m src.pipeline.cli run --start-date 2024-01-01 --end-date 2024-01-31 --force-refresh

Environment Variables:
  ENTSOE_API_KEY    API key for ENTSO-E Transparency Platform
                    Get your key at: https://transparency.entsoe.eu/
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
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
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration only")
    validate_parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to configuration directory"
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
    
    market = config.get("market", {}).get("code", "unknown")
    logger.info(f"Market: {market}")
    
    # Check for API key
    import os
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        logger.warning(
            "⚠️  ENTSOE_API_KEY not set. Will attempt to use cached data or SMARD fallback."
        )
        logger.warning(
            "   Get your API key at: https://transparency.entsoe.eu/"
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


def _handle_validate(args, logger) -> int:
    """Handle the 'validate' command."""
    logger.info(f"Validating configuration in {args.config_dir}")
    
    try:
        config = load_config(args.config_dir)
        logger.info("✓ Configuration is valid")
        
        # Print summary
        market = config.get("market", {})
        logger.info(f"  Market: {market.get('code', 'N/A')} ({market.get('name', 'N/A')})")
        logger.info(f"  Timezone: {market.get('timezone', 'N/A')}")
        
        return 0
    
    except Exception as e:
        logger.error(f"✗ Configuration invalid: {e}")
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
