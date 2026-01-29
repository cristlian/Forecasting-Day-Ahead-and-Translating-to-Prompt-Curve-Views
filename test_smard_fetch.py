"""Test script to fetch data from SMARD.de (no API key required)."""

import requests
import pandas as pd
from datetime import datetime, timedelta

def test_smard_prices():
    """Test fetching day-ahead prices from SMARD.de"""
    print("=" * 60)
    print("Testing SMARD.de Day-Ahead Prices Fetch")
    print("=" * 60)
    
    # SMARD module 8004004 = Day-ahead spot prices Germany/Luxembourg
    base_url = "https://www.smard.de/app/chart_data/8004004/DE/8004004_DE"
    
    # Test with a recent week
    test_date = datetime(2024, 12, 1)
    week_start = test_date - timedelta(days=test_date.weekday())
    timestamp_ms = int(week_start.timestamp() * 1000)
    
    url = f"{base_url}_{timestamp_ms}.json"
    print(f"\nFetching from: {url[:60]}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        series = data.get("series", [])
        records = []
        for point in series:
            ts_ms, value = point
            if value is not None:
                ts = pd.Timestamp(ts_ms, unit='ms', tz='Europe/Berlin')
                records.append({"timestamp": ts, "price": value})
        
        df = pd.DataFrame(records)
        print(f"\n✅ SUCCESS! Fetched {len(df)} price records")
        print(f"\nSample data:")
        print(df.head(10).to_string())
        print(f"\nStats: Min={df['price'].min():.2f}, Max={df['price'].max():.2f}, Mean={df['price'].mean():.2f} EUR/MWh")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return False


def test_smard_load_forecast():
    """Test fetching load forecast from SMARD.de"""
    print("\n" + "=" * 60)
    print("Testing SMARD.de Load Forecast")
    print("=" * 60)
    
    # Module 4065 = Forecasted Grid Load
    base_url = "https://www.smard.de/app/chart_data/4065/DE/4065_DE"
    
    test_date = datetime(2024, 12, 1)
    week_start = test_date - timedelta(days=test_date.weekday())
    timestamp_ms = int(week_start.timestamp() * 1000)
    
    url = f"{base_url}_{timestamp_ms}.json"
    print(f"\nFetching from: {url[:60]}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        series = data.get("series", [])
        records = [(ts, val) for ts, val in series if val is not None]
        print(f"\n✅ SUCCESS! Fetched {len(records)} load forecast records")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return False


def test_smard_wind_forecast():
    """Test fetching wind forecast from SMARD.de"""
    print("\n" + "=" * 60)
    print("Testing SMARD.de Wind Forecast (Onshore + Offshore)")
    print("=" * 60)
    
    # 4067 = Wind Onshore Forecast, 4068 = Wind Offshore Forecast
    modules = {"wind_onshore": 4067, "wind_offshore": 4068}
    
    test_date = datetime(2024, 12, 1)
    week_start = test_date - timedelta(days=test_date.weekday())
    timestamp_ms = int(week_start.timestamp() * 1000)
    
    results = {}
    for name, module_id in modules.items():
        url = f"https://www.smard.de/app/chart_data/{module_id}/DE/{module_id}_DE_{timestamp_ms}.json"
        print(f"\nFetching {name}...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            series = data.get("series", [])
            records = [(ts, val) for ts, val in series if val is not None]
            results[name] = len(records)
            print(f"   ✅ {len(records)} records")
        except Exception as e:
            print(f"   ❌ {type(e).__name__}: {e}")
            results[name] = 0
    
    return all(v > 0 for v in results.values())


def test_smard_solar_forecast():
    """Test fetching solar forecast from SMARD.de"""
    print("\n" + "=" * 60)
    print("Testing SMARD.de Solar Forecast")
    print("=" * 60)
    
    # 4069 = Solar Forecast
    module_id = 4069
    
    test_date = datetime(2024, 12, 1)
    week_start = test_date - timedelta(days=test_date.weekday())
    timestamp_ms = int(week_start.timestamp() * 1000)
    
    url = f"https://www.smard.de/app/chart_data/{module_id}/DE/{module_id}_DE_{timestamp_ms}.json"
    print(f"\nFetching from: {url[:60]}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        series = data.get("series", [])
        records = [(ts, val) for ts, val in series if val is not None]
        print(f"\n✅ SUCCESS! Fetched {len(records)} solar forecast records")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return False


def test_full_date_range():
    """Test fetching a larger date range (2 months)."""
    print("\n" + "=" * 60)
    print("Testing Full Date Range Fetch (Oct-Nov 2024)")
    print("=" * 60)
    
    base_url = "https://www.smard.de/app/chart_data/8004004/DE/8004004_DE"
    start = datetime(2024, 10, 1)
    end = datetime(2024, 11, 30)
    
    all_data = []
    current = start
    weeks_fetched = 0
    
    print(f"\nFetching {start.date()} to {end.date()}...")
    
    while current <= end:
        week_start = current - timedelta(days=current.weekday())
        timestamp_ms = int(week_start.timestamp() * 1000)
        url = f"{base_url}_{timestamp_ms}.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            series = data.get("series", [])
            for ts_ms, value in series:
                if value is not None:
                    all_data.append({"timestamp": ts_ms, "price": value})
            weeks_fetched += 1
        except:
            pass
        
        current = week_start + timedelta(days=7)
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    
    print(f"\n✅ Fetched {weeks_fetched} weeks, {len(df)} unique records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return len(df) > 1000


if __name__ == "__main__":
    print("\n🔍 SMARD.de API Test Script (No API Key Required!)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    prices = test_smard_prices()
    load = test_smard_load_forecast()
    wind = test_smard_wind_forecast()
    solar = test_smard_solar_forecast()
    bulk = test_full_date_range()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Day-Ahead Prices:  {'✅ WORKING' if prices else '❌ FAILED'}")
    print(f"Load Forecast:     {'✅ WORKING' if load else '❌ FAILED'}")
    print(f"Wind Forecast:     {'✅ WORKING' if wind else '❌ FAILED'}")
    print(f"Solar Forecast:    {'✅ WORKING' if solar else '❌ FAILED'}")
    print(f"Bulk Download:     {'✅ WORKING' if bulk else '❌ FAILED'}")
    
    if all([prices, load, wind, solar]):
        print("\n✅ All tests passed! SMARD.de is ready to use.")
        print("   No API key needed - fully automated data fetching available.")
    else:
        print("\n⚠️ Some tests failed.")
