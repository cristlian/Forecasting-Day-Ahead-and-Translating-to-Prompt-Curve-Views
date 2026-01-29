# Data Sources — DE-LU (Germany/Luxembourg) Day-Ahead Power Market

This project forecasts **Day-Ahead hourly spot prices (D+1, 24 steps)** for the **DE-LU bidding zone**
using only public data and three **Day-Ahead forecasts** as drivers: **Load**, **Wind**, **Solar**.

## Source map (datasets & API endpoints)

| Name | URL | Update Frequency | Units | Timezone | Caveats |
|---|---|---|---|---|---|
| ENTSO-E Transparency Platform — Web API (base endpoint) | https://web-api.tp.entsoe.eu/api | Day-Ahead (D+1) hourly data. <!-- Update frequency not specified in TXT; verify with source. --> | — | Mixed/unspecified; normalize to UTC internally. | Requires `securityToken` (API key). Rate limits/downtime possible. |
| ENTSO-E — Day-Ahead Prices (Target) — DE-LU | https://web-api.tp.entsoe.eu/api | Day-Ahead (D+1), hourly (24 values/day). | EUR/MWh | Mixed/unspecified; normalize to UTC internally. | Negative prices possible; prefer MAE/RMSE over MAPE. Ensure correct bidding zone (DE-LU). |
| ENTSO-E — Day-Ahead Load Forecast — DE-LU | https://web-api.tp.entsoe.eu/api | Day-Ahead (D+1), hourly. | MW | Mixed/unspecified; normalize to UTC internally. | Must use **forecast** (not actual load) to avoid look-ahead bias. |
| ENTSO-E — Day-Ahead Wind Generation Forecast — DE-LU | https://web-api.tp.entsoe.eu/api | Day-Ahead (D+1), hourly. | MW | Mixed/unspecified; normalize to UTC internally. | TXT expects aggregated onshore+offshore; confirm aggregation behavior in endpoint. |
| ENTSO-E — Day-Ahead Solar Generation Forecast — DE-LU | https://web-api.tp.entsoe.eu/api | Day-Ahead (D+1), hourly. | MW | Mixed/unspecified; normalize to UTC internally. | Confirm PV-only vs broader solar categorization in endpoint. |
| SMARD (Bundesnetzagentur) — chart_data API (fallback) | https://www.smard.de/app/chart_data | <!-- Update frequency not specified; verify per time series. --> | Varies by filter/time series. | Not specified; verify and normalize to UTC if needed. | No auth required. API requires selecting filter IDs and region/resolution. Availability of day-ahead **forecasts** (vs actuals) is not specified in TXT; verify suitability. |
| SMARD — manual data download (fallback) | https://www.smard.de/en/downloadcenter/download-market-data | <!-- Update frequency not specified; verify per download. --> | Varies | Varies | Manual download is less reproducible; must version downloaded files explicitly. |

### Notes
- The frozen scope explicitly warns about timezone confusion (UTC vs CET/CEST). The pipeline should convert all timestamps to UTC immediately on ingestion.
- If ENTSO-E token acquisition or uptime becomes a blocker, SMARD is the stated fallback; confirm that the chosen SMARD series are *forecasts* (not only actuals) before using it for features.
