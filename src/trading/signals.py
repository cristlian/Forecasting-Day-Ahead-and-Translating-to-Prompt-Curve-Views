"""Generate trading signals from forecasts."""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_clean_spark_spread(
    power_price: pd.Series,
    gas_price: float = 35.0,
    carbon_price: float = 50.0,
    heat_rate: float = 2.0,
    carbon_intensity: float = 0.4
) -> pd.Series:
    """
    Calculate Clean Spark Spread (CSS) for gas-fired generation.
    
    Formula: CSS = Power Price - (Gas Price × Heat Rate) - (Carbon Price × Carbon Intensity)
    
    For a typical CCGT:
    - Efficiency ~50% → Heat Rate = 1/0.5 = 2.0 MWh_gas/MWh_elec
    - Carbon Intensity ~0.4 tCO2/MWh_elec
    
    Args:
        power_price: Series of power prices (EUR/MWh)
        gas_price: Gas price (EUR/MWh_th) - Default: 35.0 (TTF hub proxy)
        carbon_price: EU ETS carbon price (EUR/tCO2) - Default: 50.0
        heat_rate: Gas heat rate (MWh_gas/MWh_elec) - Default: 2.0 (~50% efficiency)
        carbon_intensity: CO2 emissions (tCO2/MWh_elec) - Default: 0.4
    
    Returns:
        Series of Clean Spark Spread values (EUR/MWh)
    """
    fuel_cost = gas_price * heat_rate
    carbon_cost = carbon_price * carbon_intensity
    return power_price - fuel_cost - carbon_cost


def generate_signals(
    forecasts: pd.DataFrame,
    market_prices: Optional[pd.DataFrame] = None,
    threshold: float = 5.0,
    valid_mask: Optional[pd.Series] = None,
    gas_price: float = 35.0,
    carbon_price: float = 50.0,
    heat_rate: float = 2.0,
    carbon_intensity: float = 0.4
) -> pd.DataFrame:
    """
    Generate trading signals from forecasts.
    
    Args:
        forecasts: DataFrame with 'predicted_price' column
        market_prices: Optional current market/prompt prices for comparison
        threshold: Signal threshold (EUR/MWh)
        valid_mask: Optional mask of valid forecasts
        gas_price: Gas price (EUR/MWh_th)
        carbon_price: EU ETS carbon price (EUR/tCO2)
        heat_rate: Gas plant heat rate
        carbon_intensity: Gas plant CO2 intensity
    
    Returns:
        DataFrame with trading signals
    """
    logger.info("Generating trading signals")
    
    signals = forecasts.copy()
    
    # 1. Clean Spark Spread Calculation (The Core Economics)
    signals["clean_spark_spread"] = calculate_clean_spark_spread(
        signals["predicted_price"], 
        gas_price=gas_price, 
        carbon_price=carbon_price,
        heat_rate=heat_rate,
        carbon_intensity=carbon_intensity
    )
    
    # Calculate fuel + carbon cost for reporting
    signals["marginal_cost"] = gas_price * heat_rate + carbon_price * carbon_intensity
    
    # If market prices available, compare forecast to market (Original Plan)
    if market_prices is not None:
        signals["market_price"] = market_prices["price"]
        signals["price_spread"] = signals["predicted_price"] - signals["market_price"]
    else:
        # Otherwise use historical average as benchmark
        signals["market_price"] = signals["predicted_price"].rolling(168, min_periods=1).mean()
        signals["price_spread"] = signals["predicted_price"] - signals["market_price"]
    
    # Generate signals based on CSS (primary) and price spread (secondary)
    signals["signal"] = "HOLD"
    
    # If Price is significantly higher than market expectations -> BULLISH
    signals.loc[signals["price_spread"] > threshold, "signal"] = "BUY"
    signals.loc[signals["price_spread"] < -threshold, "signal"] = "SELL"
    
    # Margin-based dispatch signal (the key trading insight)
    signals["margin_signal"] = "OFF"
    signals.loc[signals["clean_spark_spread"] > 0, "margin_signal"] = "DISPATCH"
    
    # Calculate theoretical P&L if we dispatched based on CSS
    signals["dispatch_pnl"] = signals["clean_spark_spread"].clip(lower=0)
    
    # Apply validity mask
    if valid_mask is not None:
        signals.loc[~valid_mask, "signal"] = "INVALID"
        signals.loc[~valid_mask, "margin_signal"] = "INVALID"
        signals.loc[~valid_mask, "dispatch_pnl"] = 0
    
    # Summary
    signal_counts = signals["signal"].value_counts()
    logger.info(f"Signal distribution: {signal_counts.to_dict()}")
    
    return signals


def generate_bucket_signals(
    bucket_forecasts: pd.DataFrame,
    threshold: float = 5.0,
    gas_price: float = 35.0,
    carbon_price: float = 50.0,
    heat_rate: float = 2.0,
    carbon_intensity: float = 0.4
) -> pd.DataFrame:
    """
    Generate signals at the bucket level for block trading.
    
    Args:
        bucket_forecasts: DataFrame with bucket-level forecasts
        threshold: Signal threshold (EUR/MWh)
        gas_price: Gas price (EUR/MWh_th)
        carbon_price: EU ETS carbon price (EUR/tCO2)
        heat_rate: Gas plant heat rate
        carbon_intensity: Gas plant CO2 intensity
    
    Returns:
        DataFrame with bucket-level signals
    """
    logger.info("Generating bucket-level signals")
    
    signals = bucket_forecasts.copy()
    
    # Calculate CSS for buckets
    signals["clean_spark_spread"] = calculate_clean_spark_spread(
        signals["predicted_price"],
        gas_price=gas_price,
        carbon_price=carbon_price,
        heat_rate=heat_rate,
        carbon_intensity=carbon_intensity
    )
    
    # Marginal cost for reference
    signals["marginal_cost"] = gas_price * heat_rate + carbon_price * carbon_intensity
    
    # Generate margin-based signals
    signals["margin_signal"] = "OFF"
    signals.loc[signals["clean_spark_spread"] > 0, "margin_signal"] = "DISPATCH"
    
    # For bucket trading, compare to overall average price
    avg_price = signals["predicted_price"].mean()
    signals["price_vs_avg"] = signals["predicted_price"] - avg_price
    
    # Trading signal (buy high-value buckets, sell low-value)
    signals["signal"] = "HOLD"
    signals.loc[signals["price_vs_avg"] > threshold, "signal"] = "BUY"
    signals.loc[signals["price_vs_avg"] < -threshold, "signal"] = "SELL"
    
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
    output_path: Path,
    gas_price: float = 35.0,
    carbon_price: float = 50.0,
    heat_rate: float = 2.0,
    carbon_intensity: float = 0.4
):
    """
    Generate a markdown report of trading signals with economics.
    
    Args:
        signals: DataFrame with signals
        output_path: Path to save report
        gas_price: Gas price assumption for documentation
        carbon_price: Carbon price assumption
        heat_rate: Heat rate assumption
        carbon_intensity: Carbon intensity assumption
    """
    marginal_cost = gas_price * heat_rate + carbon_price * carbon_intensity
    
    lines = [
        "# Trading Signals Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Period:** {signals.index.min()} to {signals.index.max()}",
        f"**Hours:** {len(signals)}",
        "",
        "## Assumptions",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Gas Price (TTF) | €{gas_price:.2f}/MWh_th |",
        f"| Carbon Price (EUA) | €{carbon_price:.2f}/tCO2 |",
        f"| Heat Rate (CCGT) | {heat_rate:.2f} MWh_th/MWh_el |",
        f"| Carbon Intensity | {carbon_intensity:.2f} tCO2/MWh_el |",
        f"| **Marginal Cost** | **€{marginal_cost:.2f}/MWh_el** |",
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
    
    # Economics (Clean Spark Spread)
    if "clean_spark_spread" in signals.columns:
        avg_css = signals["clean_spark_spread"].mean()
        positive_css = (signals["clean_spark_spread"] > 0).sum()
        total_dispatch_pnl = signals.get("dispatch_pnl", signals["clean_spark_spread"].clip(lower=0)).sum()
        
        lines.append("## Economics (Margins)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Average Clean Spark Spread | €{avg_css:.2f}/MWh |")
        lines.append(f"| Profitable Hours (CSS > 0) | {positive_css} ({positive_css/len(signals)*100:.1f}%) |")
        lines.append(f"| Total Dispatch Profit (1MW) | €{total_dispatch_pnl:.2f} |")
        lines.append(f"| Avg Profit per Dispatch Hour | €{total_dispatch_pnl/max(positive_css,1):.2f}/MWh |")
        lines.append("")
        
        # CSS distribution
        lines.append("### CSS Distribution")
        lines.append("")
        lines.append("| Percentile | CSS (€/MWh) |")
        lines.append("|------------|-------------|")
        for pct in [10, 25, 50, 75, 90]:
            val = signals["clean_spark_spread"].quantile(pct/100)
            lines.append(f"| {pct}th | {val:.2f} |")
        lines.append("")

    # BUY signals
    buy_signals = signals[signals["signal"] == "BUY"]
    if len(buy_signals) > 0:
        lines.append("## BUY Signals (Long Power)")
        lines.append("")
        lines.append(f"**Count:** {len(buy_signals)}")
        lines.append(f"**Average Price Spread:** €{buy_signals['price_spread'].mean():.2f}/MWh")
        if "clean_spark_spread" in signals.columns:
            lines.append(f"**Average CSS:** €{buy_signals['clean_spark_spread'].mean():.2f}/MWh")
        lines.append("")
    
    # SELL signals
    sell_signals = signals[signals["signal"] == "SELL"]
    if len(sell_signals) > 0:
        lines.append("## SELL Signals (Short Power)")
        lines.append("")
        lines.append(f"**Count:** {len(sell_signals)}")
        lines.append(f"**Average Price Spread:** €{sell_signals['price_spread'].mean():.2f}/MWh")
        if "clean_spark_spread" in signals.columns:
            lines.append(f"**Average CSS:** €{sell_signals['clean_spark_spread'].mean():.2f}/MWh")
        lines.append("")
    
    # Backtesting against actuals (if available)
    if "actual_price" in signals.columns:
        lines.append("## Backtest (vs Actuals)")
        lines.append("")
        
        # Calculate actual CSS
        actual_css = signals["actual_price"] - marginal_cost
        actual_profitable = (actual_css > 0).sum()
        
        # Forecast accuracy for dispatch decisions
        correct_dispatch = ((signals["clean_spark_spread"] > 0) & (actual_css > 0)).sum()
        wrong_dispatch = ((signals["clean_spark_spread"] > 0) & (actual_css <= 0)).sum()
        missed_opportunity = ((signals["clean_spark_spread"] <= 0) & (actual_css > 0)).sum()
        
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Actual Profitable Hours | {actual_profitable} ({actual_profitable/len(signals)*100:.1f}%) |")
        lines.append(f"| Correct Dispatch Signals | {correct_dispatch} |")
        lines.append(f"| Wrong Dispatch (would have lost) | {wrong_dispatch} |")
        lines.append(f"| Missed Opportunities | {missed_opportunity} |")
        lines.append(f"| Dispatch Accuracy | {correct_dispatch/(correct_dispatch+wrong_dispatch)*100:.1f}% |")
        lines.append("")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Signal report saved to {output_path}")
