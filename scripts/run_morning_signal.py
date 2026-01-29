"""
Step 9: Automated Morning Trading Signal (Agent-Based Workflow)
================================================================

This script runs the Trading Agent to generate a morning execution strategy.
The agent acts as a Senior Power Trader, providing 3-bullet actionable signals.

Usage:
    python scripts/run_morning_signal.py [--no-llm] [--output-dir PATH]

Outputs:
    - reports/trading/morning_signal_YYYYMMDD_HHMMSS.json  (full data)
    - reports/trading/morning_signal_YYYYMMDD_HHMMSS.md    (formatted report)
    - reports/trading/LATEST_MORNING_SIGNAL.md             (latest signal)
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Ensure src is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
import yaml

from trading.agent import generate_morning_signal
from trading.signals import generate_signals, generate_bucket_signals
from trading.prompt_translation import translate_to_prompt_buckets

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# Market assumptions (consistent with Step 8)
GAS_PRICE = 35.0          # EUR/MWh_th
CARBON_PRICE = 50.0       # EUR/tCO2
HEAT_RATE = 2.0           # MWh_th/MWh_el
CARBON_INTENSITY = 0.4    # tCO2/MWh_el


def load_config() -> dict:
    """Load pipeline configuration."""
    config = {}
    config_dir = repo_root / "config"
    for yaml_file in config_dir.glob("*.yaml"):
        with open(yaml_file, "r") as f:
            config[yaml_file.stem] = yaml.safe_load(f)
    return config


def load_latest_predictions() -> pd.DataFrame:
    """Load the most recent model predictions."""
    preds_dir = repo_root / "outputs" / "preds_model"
    pred_files = sorted(list(preds_dir.glob("*.csv")))
    
    if not pred_files:
        raise FileNotFoundError("No prediction files found in outputs/preds_model/")
    
    latest_path = pred_files[-1]
    logger.info(f"Loading predictions from {latest_path.name}")
    
    df = pd.read_csv(latest_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.rename(columns={
        "predicted": "predicted_price", 
        "actual": "actual_price"
    }, inplace=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Morning Trading Signal using AI Trading Agent"
    )
    parser.add_argument(
        "--no-llm", 
        action="store_true",
        help="Use rule-based fallback instead of LLM"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "reports" / "trading",
        help="Output directory for signals"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🤖 TRADING AGENT: Morning Signal Generator")
    logger.info("=" * 60)
    
    # Load config
    config = load_config()
    
    # Optionally disable LLM
    if args.no_llm:
        logger.info("LLM disabled via --no-llm flag")
        config["reporting"]["llm_settings"]["enabled"] = False
    
    # Load predictions
    try:
        df = load_latest_predictions()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Run 'python -m pipeline train' first to generate predictions")
        sys.exit(1)
    
    # Use last 168 hours (1 week) as forecast period
    if "split" in df.columns:
        forecast_df = df[df["split"] == "test"].copy()
        if forecast_df.empty:
            forecast_df = df.tail(168).copy()
    else:
        forecast_df = df.tail(168).copy()
    
    logger.info(f"Forecast period: {forecast_df.index.min()} to {forecast_df.index.max()}")
    logger.info(f"Hours: {len(forecast_df)}")
    
    # Generate hourly signals (reuse Step 8 logic)
    signals_df = generate_signals(
        forecast_df,
        threshold=10.0,
        gas_price=GAS_PRICE,
        carbon_price=CARBON_PRICE,
        heat_rate=HEAT_RATE,
        carbon_intensity=CARBON_INTENSITY
    )
    
    # Generate bucket view
    bucket_view = translate_to_prompt_buckets(forecast_df)
    bucket_signals = generate_bucket_signals(
        bucket_view,
        threshold=10.0,
        gas_price=GAS_PRICE,
        carbon_price=CARBON_PRICE,
        heat_rate=HEAT_RATE,
        carbon_intensity=CARBON_INTENSITY
    )
    
    # Run the Trading Agent
    logger.info("-" * 60)
    result = generate_morning_signal(
        signals_df=signals_df,
        bucket_df=bucket_signals,
        config=config,
        output_dir=args.output_dir,
    )
    
    # Print strategy to console
    logger.info("-" * 60)
    logger.info("📊 MORNING EXECUTION STRATEGY")
    logger.info("-" * 60)
    print("\n" + result["strategy"] + "\n")
    
    logger.info("-" * 60)
    logger.info(f"✅ Signal saved to: {args.output_dir / 'LATEST_MORNING_SIGNAL.md'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
