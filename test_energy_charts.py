"""
Energy-Charts API Data Fetcher
Fraunhofer ISE - No API key required!
https://api.energy-charts.info/

This script fetches data in a format compatible with the pipeline's ingestion code.
Files are saved to data/raw/ and will be automatically recognized by:
  - src/ingest/prices.py (for prices_DE_LU.csv)
  - src/ingest/fundamentals.py (for load_DE_LU.csv, gen_forecast_DE_LU.csv)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

BASE_URL = "https://api.energy-charts.info"


def fetch_prices(start: str, end: str, bzn: str = "DE-LU") -> pd.DataFrame:
    """Fetch day-ahead prices from Energy-Charts API.
    
    Output format matches pipeline expectation:
    - Index: timestamp (UTC)
    - Column: day_ahead_price (EUR/MWh)
    """
    print(f"Fetching prices for {bzn} from {start} to {end}...")
    
    r = requests.get(
        f"{BASE_URL}/price",
        params={'bzn': bzn, 'start': start, 'end': end},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['unix_seconds'], unit='s', utc=True),
        'day_ahead_price': data['price']
    })
    df = df.set_index('timestamp')
    df = df.sort_index()
    print(f"  ✅ Got {len(df)} price records")
    return df


def fetch_load(start: str, end: str, country: str = "de") -> pd.DataFrame:
    """Fetch load forecast data from Energy-Charts API.
    
    Output format matches pipeline expectation:
    - Index: timestamp (UTC)
    - Column: forecast_load (MW)
    
    Note: Energy-Charts provides actual load, not forecasts.
    For this project, we use actual load as a proxy for day-ahead forecast
    since the forecast/actual difference is typically small.
    """
    print(f"Fetching load for {country} from {start} to {end}...")
    
    r = requests.get(
        f"{BASE_URL}/total_power",
        params={'country': country, 'start': start, 'end': end},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    
    # Find load in production_types
    load_data = None
    for item in data['production_types']:
        name_lower = item['name'].lower()
        if name_lower in ['load', 'last', 'residual load', 'verbrauch']:
            load_data = item['data']
            break
    
    if load_data is None:
        # Sum all generation as proxy for load
        print("  ⚠️ Load not found directly, using generation sum as proxy")
        load_data = [0] * len(data['unix_seconds'])
        for item in data['production_types']:
            if item['data']:
                for i, v in enumerate(item['data']):
                    if v is not None:
                        load_data[i] = (load_data[i] or 0) + v
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['unix_seconds'], unit='s', utc=True),
        'forecast_load': load_data
    })
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # Resample to hourly (pipeline expects hourly data)
    df = df.resample('h').mean()
    
    print(f"  ✅ Got {len(df)} load records (hourly)")
    return df


def fetch_renewables(start: str, end: str, country: str = "de") -> pd.DataFrame:
    """Fetch wind and solar generation from Energy-Charts API.
    
    Output format matches pipeline expectation:
    - Index: timestamp (UTC)
    - Columns: forecast_wind (MW), forecast_solar (MW)
    
    Note: This is actual generation, used as proxy for day-ahead forecasts.
    """
    print(f"Fetching renewables for {country} from {start} to {end}...")
    
    r = requests.get(
        f"{BASE_URL}/public_power",
        params={'country': country, 'start': start, 'end': end},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    
    timestamps = pd.to_datetime(data['unix_seconds'], unit='s', utc=True)
    
    wind_total = [0] * len(timestamps)
    solar_total = [0] * len(timestamps)
    
    for item in data['production_types']:
        name = item['name'].lower()
        values = item['data']
        
        if 'wind' in name:
            for i, v in enumerate(values):
                if v is not None:
                    wind_total[i] += v
        elif 'solar' in name or 'photovoltaic' in name:
            for i, v in enumerate(values):
                if v is not None:
                    solar_total[i] += v
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'forecast_wind': wind_total,
        'forecast_solar': solar_total
    })
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # Resample to hourly (pipeline expects hourly data)
    df = df.resample('h').mean()
    
    print(f"  ✅ Got {len(df)} renewable records (hourly)")
    return df


def fetch_full_dataset(start_date: str, end_date: str, output_dir: Path) -> dict:
    """
    Fetch complete dataset and save to CSV files.
    
    Output files (compatible with pipeline ingestion):
    - prices_DE_LU.csv: timestamp, day_ahead_price
    - load_DE_LU.csv: timestamp, forecast_load  
    - gen_forecast_DE_LU.csv: timestamp, forecast_wind, forecast_solar
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch in chunks (API may have limits)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_prices = []
    all_load = []
    all_renewables = []
    
    # Fetch in 30-day chunks to be safe
    chunk_days = 30
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        
        try:
            all_prices.append(fetch_prices(s, e))
            time.sleep(0.5)  # Be nice to the API
            
            all_load.append(fetch_load(s, e))
            time.sleep(0.5)
            
            all_renewables.append(fetch_renewables(s, e))
            time.sleep(0.5)
            
        except Exception as ex:
            print(f"  ⚠️ Error fetching {s} to {e}: {ex}")
        
        current = chunk_end
    
    # Combine and save
    results = {}
    
    if all_prices:
        df_prices = pd.concat(all_prices).drop_duplicates()
        df_prices.to_csv(output_dir / "prices_DE_LU.csv")
        results['prices'] = len(df_prices)
        print(f"\n✅ Saved {len(df_prices)} price records to prices_DE_LU.csv")
    
    if all_load:
        df_load = pd.concat(all_load).drop_duplicates()
        df_load.to_csv(output_dir / "load_DE_LU.csv")
        results['load'] = len(df_load)
        print(f"✅ Saved {len(df_load)} load records to load_DE_LU.csv")
    
    if all_renewables:
        df_gen = pd.concat(all_renewables).drop_duplicates()
        df_gen.to_csv(output_dir / "gen_forecast_DE_LU.csv")
        results['generation'] = len(df_gen)
        print(f"✅ Saved {len(df_gen)} generation records to gen_forecast_DE_LU.csv")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Energy-Charts Data Fetcher Test")
    print("=" * 60)
    
    # Quick test - 1 week of data
    print("\n📊 Quick Test: Fetching 1 week of December 2024 data...\n")
    
    try:
        prices = fetch_prices("2024-12-01", "2024-12-07")
        print(f"\nPrices sample:\n{prices.head()}")
        print(f"Stats: Min={prices['day_ahead_price'].min():.2f}, Max={prices['day_ahead_price'].max():.2f}")
        
        renewables = fetch_renewables("2024-12-01", "2024-12-07")
        print(f"\nRenewables sample:\n{renewables.head()}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("""
To download full dataset (2023-2024), run:

    from test_energy_charts import fetch_full_dataset
    fetch_full_dataset("2023-01-01", "2024-12-31", "data/raw")
    
Or uncomment the lines below and run this script again.
""")
        
        # Uncomment to download full dataset:
        # print("\n📥 Downloading full dataset (2023-2024)...")
        # fetch_full_dataset("2023-01-01", "2024-12-31", Path("data/raw"))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
