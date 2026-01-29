"""
Step 8: Trading Signal Generation (Improved)
--------------------------------------------
This script implements the trading signal generation and prompt curve translation.
It includes the "Trading Relevance" Pivot focusing on Clean Spark Spreads (CSS).

Key Economics:
- Clean Spark Spread = Power Price - (Gas × Heat Rate) - (Carbon × Intensity)
- Dispatch when CSS > 0 (profitable generation)
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Ensure src is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from trading.signals import generate_signals, generate_bucket_signals, save_signals, generate_signal_report
from trading.prompt_translation import translate_to_prompt_buckets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Market Assumptions (key parameters)
GAS_PRICE = 35.0          # EUR/MWh_th (TTF hub)
CARBON_PRICE = 50.0       # EUR/tCO2 (EU ETS)
HEAT_RATE = 2.0           # MWh_th/MWh_el (50% efficiency CCGT)
CARBON_INTENSITY = 0.4    # tCO2/MWh_el (natural gas CCGT)


def main():
    # 1. Load latest predictions
    preds_dir = repo_root / "outputs" / "preds_model"
    pred_files = sorted(list(preds_dir.glob("*.csv")))
    if not pred_files:
        logger.error("No prediction files found.")
        sys.exit(1)
        
    latest_pred_path = pred_files[-1]
    logger.info(f"Loading predictions from {latest_pred_path}")
    
    df = pd.read_csv(latest_pred_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.rename(columns={"predicted": "predicted_price", "actual": "actual_price"}, inplace=True)
    
    # 2. Filter for "Future" / Simulation period (Test set or OOS)
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        if test_df.empty:
            logger.warning("Test split empty, using last 7 days of data as proxy for 'Next Week' forecast.")
            test_df = df.tail(168).copy()
    else:
        test_df = df.tail(168).copy()
    
    logger.info(f"Generating signals for period: {test_df.index.min()} to {test_df.index.max()}")
    
    # 3. Generate Hourly Signals with Clean Spark Spread
    signals = generate_signals(
        test_df,
        threshold=10.0,
        gas_price=GAS_PRICE,
        carbon_price=CARBON_PRICE,
        heat_rate=HEAT_RATE,
        carbon_intensity=CARBON_INTENSITY
    )
    
    # 4. Prompt Curve View (Buckets)
    bucket_view = translate_to_prompt_buckets(test_df)
    
    # 5. Generate Bucket Signals
    bucket_signals = generate_bucket_signals(
        bucket_view,
        threshold=10.0,
        gas_price=GAS_PRICE,
        carbon_price=CARBON_PRICE,
        heat_rate=HEAT_RATE,
        carbon_intensity=CARBON_INTENSITY
    )
    
    # 6. Save Outputs
    output_dir = repo_root / "reports" / "trading"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_signals(signals, output_dir / "hourly_signals.csv")
    bucket_signals.to_csv(output_dir / "prompt_view.csv")
    logger.info(f"Bucket view saved to {output_dir / 'prompt_view.csv'}")
    
    # 7. Generate Report
    generate_signal_report(
        signals, 
        output_dir / "signal_report.md",
        gas_price=GAS_PRICE,
        carbon_price=CARBON_PRICE,
        heat_rate=HEAT_RATE,
        carbon_intensity=CARBON_INTENSITY
    )
    
    # 8. Create Deliverable: Trading Guidance
    marginal_cost = GAS_PRICE * HEAT_RATE + CARBON_PRICE * CARBON_INTENSITY
    avg_css = signals["clean_spark_spread"].mean()
    profitable_hours = (signals["clean_spark_spread"] > 0).sum()
    total_pnl = signals["dispatch_pnl"].sum()
    
    guidance_content = f"""# Trading Guidance: Forecast to Execution

## 1. Trading Relevance Pivot: Margins over Price

Instead of purely trading on price direction, we focus on **Clean Spark Spreads (CSS)** 
to capture the fundamental economics of gas-fired generation.

### Formula (CCGT)

$$
\\text{{CSS}} = \\text{{Power Price}} - (\\text{{Gas Price}} \\times \\text{{Heat Rate}}) - (\\text{{Carbon Price}} \\times \\text{{Carbon Intensity}})
$$

### Current Assumptions

| Parameter | Value | Source/Rationale |
|-----------|-------|------------------|
| Gas Price (TTF) | €{GAS_PRICE:.2f}/MWh_th | European hub benchmark |
| Carbon Price (EUA) | €{CARBON_PRICE:.2f}/tCO2 | EU ETS market |
| Heat Rate | {HEAT_RATE:.2f} MWh_th/MWh_el | ~50% efficiency (modern CCGT) |
| Carbon Intensity | {CARBON_INTENSITY:.2f} tCO2/MWh_el | Natural gas combustion |
| **Marginal Cost** | **€{marginal_cost:.2f}/MWh_el** | Fuel + Carbon |

## 2. Current Period Results

**Analysis Period:** {signals.index.min().strftime('%Y-%m-%d')} to {signals.index.max().strftime('%Y-%m-%d')}

| Metric | Value |
|--------|-------|
| Total Hours Analyzed | {len(signals)} |
| Average CSS | €{avg_css:.2f}/MWh |
| Profitable Dispatch Hours | {profitable_hours} ({profitable_hours/len(signals)*100:.1f}%) |
| Total Dispatch Profit (1MW plant) | €{total_pnl:.2f} |

## 3. Signal Logic

### Margin-Based Dispatch (Primary)
- **DISPATCH:** CSS > 0 → Gas plant generation is profitable
- **OFF:** CSS ≤ 0 → Generation is uneconomic, buy from market instead

### Price Direction (Secondary)  
- **BUY:** Forecast significantly above rolling average → Market tightness expected
- **SELL:** Forecast significantly below average → Oversupply expected
- **HOLD:** Price within normal range

## 4. Invalidation Rules (Risk Management)

Trading signals should be invalidated or position sizes reduced when:

| Rule | Threshold | Action |
|------|-----------|--------|
| Forecast Drift | >10% vs previous run | Reduce position 50% |
| Wind/Solar Actuals vs Forecast | >2GW deviation | Invalidate DA signals |
| Persistent Negative CSS | >6 consecutive hours | Switch to optionality mode |
| Extreme Price Volatility | Hourly σ > €30/MWh | Widen entry thresholds |

## 5. Prompt View Usage (Block Trading)

The aggregated bucket view supports standardized block trades:

| Bucket | Typical Profile | Trading Focus |
|--------|-----------------|---------------|
| **Peak** (08-20 weekdays) | High demand, high price | Spread capture, CSS arbitrage |
| **Off-Peak Night** (00-06) | Low demand | Wind-driven negative prices |
| **Off-Peak Late** (22-24) | Transition period | Position for next day |
| **Shoulder** (06-08, 20-22) | Volatile | Short-term trades |
| **Weekend** | Variable, often low | Battery/storage optimization |

## 6. Execution Recommendations

Based on current analysis:

1. **Baseload Generation:** {"PROFITABLE - dispatch base capacity" if avg_css > 0 else "UNECONOMIC - minimize generation"}
2. **Average Margin:** €{avg_css:.2f}/MWh {"(healthy)" if avg_css > 20 else "(marginal)" if avg_css > 0 else "(negative)"}
3. **Risk Level:** {"LOW - consistent profitability" if profitable_hours/len(signals) > 0.8 else "MEDIUM - mixed signals" if profitable_hours/len(signals) > 0.5 else "HIGH - frequent losses"}

---
*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_dir / "guidance.md", "w", encoding='utf-8') as f:
        f.write(guidance_content)
    logger.info(f"Guidance document saved to {output_dir / 'guidance.md'}")


if __name__ == "__main__":
    main()
