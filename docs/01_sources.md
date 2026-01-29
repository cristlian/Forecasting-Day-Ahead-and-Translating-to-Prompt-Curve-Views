# Step 1: Data Sources

## Primary Data Sources

### 1. Day-Ahead Prices
**Source:** Energy-Charts API (Fraunhofer ISE)  
**Endpoint:** https://api.energy-charts.info/price  
**Frequency:** Hourly  
**Coverage:** 2015-present  
**Notes:**
- No API key required
- Delivered in UTC, easy to normalize
- High-quality Germany/Luxembourg coverage

### 2. Load (Forecast Proxy)
**Source:** Energy-Charts API  
**Endpoint:** https://api.energy-charts.info/total_power  
**Frequency:** Hourly  
**Notes:**
- Provides actual load; used as a proxy for day-ahead forecast
- Stable and reliable data quality

### 3. Wind Generation
**Source:** Energy-Charts API  
**Endpoint:** https://api.energy-charts.info/public_power  
**Frequency:** Hourly  
**Notes:**
- Aggregates wind onshore/offshore

### 4. Solar Generation
**Source:** Energy-Charts API  
**Endpoint:** https://api.energy-charts.info/public_power  
**Frequency:** Hourly  

### 5. Temperature (Optional)
**Source:** Open-Meteo API or ERA5  
**Frequency:** Hourly  
**Coverage:** Major cities in market zone  
**Notes:**
- Use as proxy for heating/cooling demand

### 6. Fuel Prices (Optional Enhancement)
**Source:** Manual collection or ICE/EEX data  
**Gas:** TTF natural gas prices  
**Coal:** API2 coal prices  
**Notes:**
- Often daily frequency only
- May need forward-fill for hourly model

## Alternative Sources (Fallback)
- **SMARD.de (Bundesnetzagentur):** No API key required, Germany-specific
- **Open Power System Data (OPSD):** Aggregated historical data

## Data Access Notes
- Energy-Charts and SMARD require no API key
- Recommend caching raw downloads
