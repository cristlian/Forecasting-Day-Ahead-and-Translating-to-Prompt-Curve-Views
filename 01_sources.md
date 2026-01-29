# Data Sources — DE-LU (Germany/Luxembourg) Day-Ahead Power Market

This project forecasts **Day-Ahead hourly spot prices (D+1, 24 steps)** for the **DE-LU bidding zone**
using only public data and three **Day-Ahead forecasts** as drivers: **Load**, **Wind**, **Solar**.

## Source map (datasets & API endpoints)

| Name | URL | Update Frequency | Units | Timezone | Caveats |
|---|---|---|---|---|---|
| Energy-Charts API — Prices (Target) | https://api.energy-charts.info/price | Day-Ahead (D+1), hourly. | EUR/MWh | UTC | No API key required. Negative prices possible; prefer MAE/RMSE over MAPE. |
| Energy-Charts API — Load (Forecast Proxy) | https://api.energy-charts.info/total_power | Hourly | MW | UTC | Uses actual load as a proxy for day-ahead forecast. |
| Energy-Charts API — Wind/Solar | https://api.energy-charts.info/public_power | Hourly | MW | UTC | Aggregates wind onshore/offshore and solar. |
| SMARD (Bundesnetzagentur) — chart_data API (fallback) | https://www.smard.de/app/chart_data | <!-- Update frequency not specified; verify per time series. --> | Varies by filter/time series. | Not specified; normalize to UTC if needed. | No auth required. API requires selecting filter IDs and region/resolution. |
| SMARD — manual data download (fallback) | https://www.smard.de/en/downloadcenter/download-market-data | <!-- Update frequency not specified; verify per download. --> | Varies | Varies | Manual download is less reproducible; must version downloaded files explicitly. |

### Notes
- The frozen scope explicitly warns about timezone confusion (UTC vs CET/CEST). The pipeline should convert all timestamps to UTC immediately on ingestion.
- If SMARD availability becomes a blocker, use Energy-Charts as the primary data source and cache raw downloads.
