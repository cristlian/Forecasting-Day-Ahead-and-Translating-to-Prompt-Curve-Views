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

## Alternative: Manual Download (SMARD)

If you prefer manual download from SMARD.de (Bundesnetzagentur):

### 1. Setup
1. Create a `data/raw` directory in your project root if it doesn't exist.
2. Store all downloaded CSV files in this folder.

### 2. Download Day-Ahead Prices
1. Go to **SMARD** > Day-Ahead Prices (module 8004004).
2. Select Germany/Luxembourg (DE-LU) and the date range.
3. Export CSV and rename to: `prices_DE_LU.csv`
4. Place it in `data/raw/`.

### 3. Download Load Forecast
1. Go to **SMARD** > Forecasted Grid Load (module 4065).
2. Export CSV and rename to: `load_DE_LU.csv`
3. Place it in `data/raw/`.

### 4. Download Wind and Solar Forecast
1. Go to **SMARD** > Wind Onshore Forecast (module 4067) and Wind Offshore Forecast (module 4068).
2. Go to **SMARD** > Solar Forecast (module 4069).
3. Export and merge into a single CSV with columns `forecast_wind` and `forecast_solar`.
4. Rename the final file to: `gen_forecast_DE_LU.csv` and place it in `data/raw/`.

## Integration
The system is now configured to automatically look for these files in `data/raw/` before attempting any online fallback.
