# Step 3: Design Choices & Rationale

## Baseline Model Choice
**Selected:** Naive Seasonal (Same-Hour Last Week)

**Rationale:**
- Simple, interpretable benchmark
- Captures weekly seasonality (workday/weekend patterns)
- Common industry baseline for power forecasting
- Easy to beat → validates improved model value

**Alternatives Considered:**
- Same-hour yesterday: Too volatile, misses weekly patterns
- Exponential smoothing: More complex, marginal gain over naive
- Verdict: Stick with 168-hour lag baseline

## Improved Model Choice
**Selected:** LightGBM (Gradient Boosted Trees)

**Rationale:**
- Excellent performance on tabular data with mixed feature types
- Handles missing data gracefully
- Fast training and inference
- Native handling of categorical features
- Well-suited for non-linear interactions (e.g., hour × residual load)

**Alternatives Considered:**
- **XGBoost:** Equally valid, LightGBM slightly faster
- **Random Forest:** Less prone to overfitting but typically lower accuracy
- **Neural Networks (LSTM, Transformer):** Overkill for this task, harder to debug, longer training
- **Linear Models:** Too simple, miss non-linear price dynamics

**Decision:** LightGBM as primary, XGBoost as fallback option in config

## Feature Engineering Philosophy
**Approach:** Domain-informed + automated feature generation

**Key Features:**
1. **Residual Load:** `load - wind - solar`
   - Most important price driver in modern markets
   
2. **Calendar Features:** hour, day-of-week, month, holidays
   - Capture demand patterns
   
3. **Lags:** 1h, 2h, 3h, 24h, 48h, 168h
   - Short-term momentum + day-ahead reference
   
4. **Rolling Statistics:** 6h, 24h, 168h windows
   - Smooth out noise, capture recent trends
   
5. **Interactions:** hour × residual_load, hour × day_of_week
   - Non-linear effects (e.g., low residual load at noon → negative prices)

**Features NOT Included (intentionally):**
- Autoregressive lags >7 days: Diminishing returns, overfitting risk
- High-frequency lags (<1h): Data is hourly, not useful
- External macro variables: Low signal for day-ahead horizon

## Cross-Validation Strategy
**Selected:** Rolling-Origin CV with Fixed Window

**Setup:**
- Train window: 365 days
- Test window: 7 days
- Number of splits: 10
- No gap between train/test (data leakage already prevented by lag features)

**Rationale:**
- Simulates production scenario (retrain weekly with 1-year history)
- Detects temporal drift in model performance
- More conservative than expanding-window CV

## QA Gate Policy
**Philosophy:** Strict gates, fail-fast

**Critical Gates:**
- >1% missing in price/load → FAIL
- Duplicate timestamps → FAIL
- >2-hour data gaps → FAIL
- >0.1% out-of-range values → FAIL

**Rationale:**
- Bad data in → bad forecasts out
- Better to stop pipeline than produce unreliable signals
- In production, would trigger alerts and manual review

## Trading Signal Translation
**Approach:** Forecast → Hourly Fair Value → Prompt Curve Buckets

**Buckets:**
- Off-peak 1: 00:00-06:00
- Off-peak 2: 22:00-24:00
- Peak: 08:00-20:00 (weekdays)
- Shoulder: All other hours

**Signal Generation:**
- If forecast > current prompt by >5 EUR/MWh → BUY signal
- If forecast < current prompt by >5 EUR/MWh → SELL signal
- Else → NO TRADE

**Invalidation Rules:**
- If prediction interval width >30 EUR/MWh → invalidate (high uncertainty)
- If key drivers missing (e.g., no wind forecast) → invalidate

**Rationale:**
- 5 EUR threshold avoids noise trading
- Bucket-level signals align with how power is actually traded
- Invalidation prevents trading on unreliable forecasts

## LLM Commentary (Optional)
**Decision:** Include framework, disable by default

**Rationale:**
- Shows awareness of AI-assisted reporting trend
- Template-based → can be easily activated with API key
- Not core to technical solution, so optional is fine

**Use Case:**
- Auto-generate "Market Commentary" section
- Summarize key drivers of price movements
- Flag anomalies or model concerns

## Code Organization Principles
- **Config-driven:** No hardcoded paths or parameters
- **Modular:** Each step is independent, testable
- **Fail-fast:** QA gates stop bad data early
- **Reproducible:** Random seeds, versioned dependencies
- **Transparent:** Logging at every step

## Trade-Offs & Limitations
1. **Hourly resolution:** Sub-hourly volatility not captured (acceptable for day-ahead focus)
2. **Single-market:** Not multi-market (would need cross-border flow features)
3. **No ensemble:** Single model for simplicity (could ensemble LightGBM + XGBoost)
4. **No online learning:** Retrain weekly, not real-time (production would need this)
5. **Fuel prices optional:** Often unavailable at hourly frequency, model works without them
