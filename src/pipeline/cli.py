"""Command-line interface for the power fair value pipeline."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .run import run_pipeline
from .config import load_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Power Fair Value Forecasting Pipeline"
    )
    
    parser.add_argument(
        "command",
        choices=["run", "validate", "backtest"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date for forecast (YYYY-MM-DD). Defaults to today."
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to configuration directory"
    )
    
    parser.add_argument(
        "--market",
        type=str,
        default=None,
        help="Override market code (e.g., DE, FR)"
    )
    
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA gates (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_dir)
    
    # Override market if specified
    if args.market:
        config["market"]["code"] = args.market
    
    # Parse date
    target_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    
    try:
        if args.command == "run":
            run_pipeline(
                config=config,
                target_date=target_date,
                skip_qa=args.skip_qa
            )
            print("✓ Pipeline completed successfully")
            return 0
        
        elif args.command == "validate":
            # TODO: Implement validation-only mode
            print("Validation mode not yet implemented")
            return 1
        
        elif args.command == "backtest":
            # TODO: Implement backtest mode
            print("Backtest mode not yet implemented")
            return 1
    
    except Exception as e:
        print(f"✗ Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
