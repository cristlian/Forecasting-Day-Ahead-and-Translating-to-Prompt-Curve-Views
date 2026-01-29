# Step 2: Schema & Data Normalization Notes

## Timezone Strategy
**Decision:** All data stored and processed in **market-local timezone** (e.g., Europe/Berlin for DE)

**Rationale:**
- Power markets operate on local time
- Price signals tied to local demand patterns
- Avoids confusion in DST transition handling

## DST Transition Handling

### Spring Forward (March)
- **Problem:** 1 hour "disappears" (e.g., 2:00 AM → 3:00 AM)
- **Solution:** **Drop** the non-existent hour from data
- **Impact:** 23-hour day in dataset
- **Model handling:** Ensure no forward-fill across DST gap

### Fall Back (October)
- **Problem:** 1 hour "repeats" (e.g., 2:00 AM → 2:00 AM again)
- **Solution:** **Average** the two 2:00 AM prices
- **Impact:** Single entry for duplicated hour
- **Alternative:** Keep first occurrence only (simpler, acceptable)

## Unit Conversions

### Power & Energy
- **Load/Generation:** Store as **MW** (megawatts)
- **Prices:** Store as **EUR/MWh** (or local currency)
- Rationale: Standard industry units

### Temperature
- **Storage:** Celsius
- **Alternative:** Could normalize to Z-scores if multi-region

## Column Naming Convention
- Use **snake_case** throughout
- Prefix temporal features: `lag_24h_price`, `rolling_7d_mean_load`
- Boolean flags: `is_weekend`, `is_holiday`
- Avoid abbreviations in schema (clarity over brevity)

## Missing Data Policy
- **Critical columns** (price, load): Fail QA if >1% missing
- **Optional columns** (weather, fuel): Tolerate up to 10% missing
- **Imputation:** Forward-fill up to 2 hours, else flag for review
- **Never backfill** to avoid look-ahead bias

## Data Type Enforcement
- Timestamps: `datetime64[ns, tz]` (timezone-aware)
- Numeric columns: `float64` (avoid precision issues)
- Validate on every load: schema.yaml contract enforced

## Duplicate Handling
- Check for duplicate timestamps in QA
- Duplicates → **fail QA gate** (likely data error)
- Exception: DST fall-back handled explicitly

## Frequency Alignment
- Target: Hourly resolution
- If sub-hourly data available (15-min), aggregate to hourly mean
- Ensure no gaps: continuous hourly index required
