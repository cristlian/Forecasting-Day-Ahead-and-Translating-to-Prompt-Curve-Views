# Data Download Guide

This guide explains how to obtain the required data for the DE-LU power market forecasting pipeline.

## Recommended: Energy-Charts API (Automated)

**No API key required!** The Energy-Charts API from Fraunhofer ISE is free and reliable.

### Quick Download (2 years of data)

```bash
# Run the data fetcher
python -c "from test_energy_charts import fetch_full_dataset; fetch_full_dataset('2023-01-01', '2024-12-31', 'data/raw')"
```

This will create:
- `data/raw/prices_DE_LU.csv` - Day-ahead prices
- `data/raw/load_DE_LU.csv` - Load data (used as forecast proxy)
- `data/raw/gen_forecast_DE_LU.csv` - Wind and solar generation

### Data Notes
- **Resolution**: Hourly (pipeline expectation)
- **Timezone**: UTC
- **Format**: CSV with timestamp index
- Energy-Charts provides actual generation data, which we use as a proxy for day-ahead forecasts

---

## Alternative: Manual Download (ENTSO-E)

If you prefer manual download from ENTSO-E Transparency Platform:

### 1. Setup
1. Create a `data/raw` directory in your project root if it doesn't exist.
2. Store all downloaded CSV files in this folder.

### 2. Download Day-Ahead Prices
**⚠️ IMPORTANT**: Day-Ahead Prices are NOT in the Transmission tab, they are in the **Market** tab!

1. Go to **Market** > **Day-ahead Prices**.
2. Select:
   - **Bidding Zone**: `DE-LU` (Germany/Luxembourg)
   - **Date Range**: `01.01.2023` to `31.12.2024` (or your full range)
   - **Note**: Max date range is 7 days, so you'll need to download multiple files and combine them
3. Click **Export** > **CSV**.
4. If downloading multiple files, combine them into one CSV file.
5. Rename the final file to: `prices_DE_LU.csv`
6. Place it in `data/raw/`.

## 3. Download Load Forecast
1. Go to **Load** > **Total Load - Day Ahead / Actual**.
2. Select:
   - **Bidding Zone**: `DE-LU`
   - **Date Range**: `01.01.2023` to `31.12.2024`
3. Click **Export** > **CSV**.
4. Rename the file to: `load_DE_LU.csv`
5. Place it in `data/raw/`.

## 4. Download Wind and Solar Forecast
1. Go to **Generation** > **Wind and Solar Generation Forecast**.
2. Select:
   - **Bidding Zone**: `DE-LU`
   - **Date Range**: `01.01.2023` to `31.12.2024`
3. Click **Export** > **CSV**.
4. Rename the file to: `gen_forecast_DE_LU.csv`
5. Place it in `data/raw/`.

## Integration
The system is now configured to automatically look for these files in `data/raw/` before attempting to use the API.
