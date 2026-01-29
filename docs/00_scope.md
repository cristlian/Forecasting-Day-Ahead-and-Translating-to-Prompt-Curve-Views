# Step 0: Project Scope & Objectives

## Executive Summary
Build a reproducible daily pipeline for **DE-LU (Germany-Luxembourg)** day-ahead hourly price forecasting using only public data.

## Market Selection
- **Market**: DE-LU (Germany-Luxembourg Bidding Zone)
- **Rationale**: High renewable penetration, excellent data quality, benchmark European market

## Target Variable
- **What**: Day-Ahead hourly spot price (EUR/MWh)
- **Horizon**: 24 hours (00:00-23:00) for Day D+1
- **Auction**: Prices determined by 12:00 CET day-ahead auction

## Fundamental Drivers (Forecasts Only)
1. **Day-Ahead Load Forecast** [MW] - Positive correlation with price
2. **Day-Ahead Wind Forecast** [MW] - Negative correlation (can drive negative prices)
3. **Day-Ahead Solar Forecast** [MW] - Negative correlation (peak hours 10-16)

## Data Sources
- **Primary**: Energy-Charts API (Fraunhofer ISE, no token required)
- **Fallback**: SMARD.de (Bundesnetzagentur, no token required)

## Success Metrics
1. **Data QA**: <1% gaps in target, correct UTC/CET handling, no look-ahead bias
2. **Forecast Accuracy**: MAE < 15 EUR/MWh, beat naive baseline by >5%
3. **Engineering**: Single-command pipeline execution

## Key Constraints
- Public data only (no Bloomberg, Refinitiv)
- All internal processing in UTC
- Only day-ahead forecasts as features (no actuals - avoid look-ahead bias)
- Handle DST transitions correctly

## Out of Scope
- Intraday/balancing markets
- Grid constraints, nodal pricing
- Live trading execution

## Date Range
- **Training**: 2023-01-01 to 2023-12-31
- **Testing**: 2024-01-01 to 2024-12-31
