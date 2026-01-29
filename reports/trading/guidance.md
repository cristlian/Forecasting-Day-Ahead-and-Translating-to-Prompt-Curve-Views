# Trading Guidance: Forecast to Execution

## 1. Trading Relevance Pivot: Margins over Price

Instead of purely trading on price direction, we focus on **Clean Spark Spreads (CSS)** 
to capture the fundamental economics of gas-fired generation.

### Formula (CCGT)

$$
\text{CSS} = \text{Power Price} - (\text{Gas Price} \times \text{Heat Rate}) - (\text{Carbon Price} \times \text{Carbon Intensity})
$$

### Current Assumptions

| Parameter | Value | Source/Rationale |
|-----------|-------|------------------|
| Gas Price (TTF) | €35.00/MWh_th | European hub benchmark |
| Carbon Price (EUA) | €50.00/tCO2 | EU ETS market |
| Heat Rate | 2.00 MWh_th/MWh_el | ~50% efficiency (modern CCGT) |
| Carbon Intensity | 0.40 tCO2/MWh_el | Natural gas combustion |
| **Marginal Cost** | **€90.00/MWh_el** | Fuel + Carbon |

## 2. Current Period Results

**Analysis Period:** 2024-10-15 to 2024-10-23

| Metric | Value |
|--------|-------|
| Total Hours Analyzed | 168 |
| Average CSS | €5.65/MWh |
| Profitable Dispatch Hours | 104 (61.9%) |
| Total Dispatch Profit (1MW plant) | €2769.86 |

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

1. **Baseload Generation:** PROFITABLE - dispatch base capacity
2. **Average Margin:** €5.65/MWh (marginal)
3. **Risk Level:** MEDIUM - mixed signals

---
*Generated: 2026-01-29 05:14:43*
