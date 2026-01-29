# Step 1: Data Sources

## Primary Data Sources

### 1. Day-Ahead Prices
**Source:** ENTSO-E Transparency Platform  
**Endpoint:** Market Documents / Day-Ahead Prices  
**Frequency:** Hourly  
**Coverage:** 2020-present  
**Notes:**
- Requires API registration (free)
- Returns prices in local timezone with DST handling
- Quality: High, authoritative source

### 2. Actual Load (Demand)
**Source:** ENTSO-E Transparency Platform  
**Endpoint:** Actual Total Load  
**Frequency:** Hourly  
**Notes:**
- Real-time data with 1-hour lag
- Need forecast load for forward-looking predictions

### 3. Wind Generation
**Source:** ENTSO-E Transparency Platform  
**Endpoint:** Actual Generation per Type (Wind Onshore + Offshore)  
**Frequency:** Hourly  
**Notes:**
- Split by onshore/offshore in some markets
- Aggregate for model simplicity

### 4. Solar Generation
**Source:** ENTSO-E Transparency Platform  
**Endpoint:** Actual Generation per Type (Solar)  
**Frequency:** Hourly  

### 5. Temperature (Optional)
**Source:** Open-Meteo API or ERA5  
**Frequency:** Hourly  
**Coverage:** Major cities in market zone  
**Notes:**
- Use as proxy for heating/cooling demand
- Free tier sufficient for historical data

### 6. Fuel Prices (Optional Enhancement)
**Source:** Manual collection or ICE/EEX data  
**Gas:** TTF natural gas prices  
**Coal:** API2 coal prices  
**Notes:**
- Often daily frequency only
- May need forward-fill for hourly model

## Alternative Sources (Fallback)
- **Open Power System Data (OPSD):** Aggregated historical data
- **Energy Charts (Fraunhofer ISE):** Germany-specific, high quality
- **EIA (US markets):** If switching to US market analysis

## Data Access Notes
- ENTSO-E requires free API token
- Rate limits: 400 requests/minute
- Historical data readily available
- Recommend caching raw downloads
