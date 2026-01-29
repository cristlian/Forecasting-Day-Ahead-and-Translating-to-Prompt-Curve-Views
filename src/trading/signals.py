"""Generate trading signals from forecasts."""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_signals(
    forecasts: pd.DataFrame,
    market_prices: Optional[pd.DataFrame] = None,
    threshold: float = 5.0,
    valid_mask: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate trading signals from forecasts.
    
    Args:
        forecasts: DataFrame with 'predicted_price' column
        market_prices: Optional current market/prompt prices for comparison
        threshold: Signal threshold (EUR/MWh)
        valid_mask: Optional mask of valid forecasts
    
    Returns:
        DataFrame with trading signals
    """
    logger.info("Generating trading signals")
    
    signals = forecasts.copy()
    
    # If market prices available, compare forecast to market
    if market_prices is not None:
        signals["market_price"] = market_prices["price"]
        signals["spread"] = signals["predicted_price"] - signals["market_price"]
    else:
        # Otherwise use historical average as benchmark
        signals["market_price"] = signals["predicted_price"].rolling(168).mean()
        signals["spread"] = signals["predicted_price"] - signals["market_price"]
    
    # Generate signals based on threshold
    signals["signal"] = "HOLD"
    signals.loc[signals["spread"] > threshold, "signal"] = "BUY"
    signals.loc[signals["spread"] < -threshold, "signal"] = "SELL"
    
    # Apply validity mask
    if valid_mask is not None:
        signals.loc[~valid_mask, "signal"] = "INVALID"
    
    # Summary
    signal_counts = signals["signal"].value_counts()
    logger.info(f"Signal distribution: {signal_counts.to_dict()}")
    
    return signals


def generate_bucket_signals(
    bucket_forecasts: pd.DataFrame,
    threshold: float = 5.0
) -> pd.DataFrame:
    """
    Generate signals at the bucket level.
    
    Args:
        bucket_forecasts: DataFrame with bucket-level forecasts
        threshold: Signal threshold (EUR/MWh)
    
    Returns:
        DataFrame with bucket-level signals
    """
    logger.info("Generating bucket-level signals")
    
    signals = bucket_forecasts.copy()
    
    # Simple comparison to historical average
    # In production, would compare to current prompt prices
    signals["historical_avg"] = signals["predicted_price"].rolling(10).mean()
    signals["spread"] = signals["predicted_price"] - signals["historical_avg"]
    
    # Generate signals
    signals["signal"] = "HOLD"
    signals.loc[signals["spread"] > threshold, "signal"] = "BUY"
    signals.loc[signals["spread"] < -threshold, "signal"] = "SELL"
    
    return signals


def save_signals(
    signals: pd.DataFrame,
    output_path: Path,
    metadata: Optional[Dict] = None
):
    """
    Save trading signals to CSV.
    
    Args:
        signals: DataFrame with signals
        output_path: Path to save signals
        metadata: Optional metadata (model version, thresholds, etc.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save signals
    signals.to_csv(output_path, index=True)
    logger.info(f"Signals saved to {output_path}")
    
    # Save metadata if provided
    if metadata:
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Signal metadata saved to {metadata_path}")


def generate_signal_report(
    signals: pd.DataFrame,
    output_path: Path
):
    """
    Generate a markdown report of trading signals.
    
    Args:
        signals: DataFrame with signals
        output_path: Path to save report
    """
    lines = [
        "# Trading Signals Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Signal Summary",
        "",
    ]
    
    # Signal distribution
    signal_counts = signals["signal"].value_counts()
    lines.append("| Signal | Count | Percentage |")
    lines.append("|--------|-------|------------|")
    
    for signal, count in signal_counts.items():
        pct = count / len(signals) * 100
        lines.append(f"| {signal} | {count} | {pct:.1f}% |")
    
    lines.append("")
    
    # BUY signals
    buy_signals = signals[signals["signal"] == "BUY"]
    if len(buy_signals) > 0:
        lines.append("## BUY Signals")
        lines.append("")
        lines.append(f"**Count:** {len(buy_signals)}")
        lines.append(f"**Average spread:** {buy_signals['spread'].mean():.2f} EUR/MWh")
        lines.append("")
    
    # SELL signals
    sell_signals = signals[signals["signal"] == "SELL"]
    if len(sell_signals) > 0:
        lines.append("## SELL Signals")
        lines.append("")
        lines.append(f"**Count:** {len(sell_signals)}")
        lines.append(f"**Average spread:** {sell_signals['spread'].mean():.2f} EUR/MWh")
        lines.append("")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Signal report saved to {output_path}")
