"""Test script to fetch data from ENTSO-E using entsoe-py."""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_entsoe_prices():
    """Test fetching day-ahead prices from ENTSO-E."""
    print("=" * 60)
    print("Testing ENTSO-E Day-Ahead Prices Fetch")
    print("=" * 60)
    
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        print("❌ ERROR: ENTSOE_API_KEY not found in environment")
        print("   Please set it in .env file or environment")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...")
    
    try:
        from entsoe import EntsoePandasClient
        print("✓ entsoe-py package imported successfully")
    except ImportError as e:
        print(f"❌ ERROR: Failed to import entsoe-py: {e}")
        print("   Install with: pip install entsoe-py")
        return False
    
    # Test parameters
    area_code = "10Y1001A1001A82H"  # DE-LU bidding zone
    start_date = pd.Timestamp("2024-12-01", tz="Europe/Berlin")
    end_date = pd.Timestamp("2024-12-07", tz="Europe/Berlin")
    
    print(f"\nFetching data for:")
    print(f"  Area: DE-LU ({area_code})")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    
    try:
        client = EntsoePandasClient(api_key=api_key)
        print("\n✓ Client initialized")
        
        print("\nAttempting to fetch day-ahead prices...")
        prices = client.query_day_ahead_prices(
            area_code, 
            start=start_date, 
            end=end_date
        )
        
        print(f"\n✅ SUCCESS! Fetched {len(prices)} price records")
        print(f"\nSample data:")
        print(prices.head(10))
        print(f"\nPrice statistics:")
        print(f"  Min: {prices.min():.2f} EUR/MWh")
        print(f"  Max: {prices.max():.2f} EUR/MWh")
        print(f"  Mean: {prices.mean():.2f} EUR/MWh")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to fetch prices")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return False


def test_entsoe_fundamentals():
    """Test fetching fundamental data from ENTSO-E."""
    print("\n" + "=" * 60)
    print("Testing ENTSO-E Fundamentals Fetch")
    print("=" * 60)
    
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        print("❌ ERROR: ENTSOE_API_KEY not found")
        return False
    
    try:
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=api_key)
        
        area_code = "10Y1001A1001A82H"
        start_date = pd.Timestamp("2024-12-01", tz="Europe/Berlin")
        end_date = pd.Timestamp("2024-12-07", tz="Europe/Berlin")
        
        results = {}
        
        # Test Load Forecast
        print("\n1. Fetching Load Forecast...")
        try:
            load = client.query_load_forecast(area_code, start=start_date, end=end_date)
            results['load'] = len(load)
            print(f"   ✅ SUCCESS: {len(load)} records")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            results['load'] = None
        
        # Test Wind & Solar Forecast
        print("\n2. Fetching Wind & Solar Forecast...")
        try:
            gen = client.query_wind_and_solar_forecast(area_code, start=start_date, end=end_date)
            results['generation'] = len(gen)
            print(f"   ✅ SUCCESS: {len(gen)} records")
            print(f"   Columns: {list(gen.columns)}")
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            results['generation'] = None
        
        success = all(v is not None for v in results.values())
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n🔍 ENTSO-E API Test Script")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test prices
    prices_ok = test_entsoe_prices()
    
    # Test fundamentals
    fundamentals_ok = test_entsoe_fundamentals()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Day-Ahead Prices:  {'✅ WORKING' if prices_ok else '❌ FAILED'}")
    print(f"Fundamentals:      {'✅ WORKING' if fundamentals_ok else '❌ FAILED'}")
    
    if prices_ok and fundamentals_ok:
        print("\n✅ All tests passed! entsoe-py is working correctly.")
        print("   You can proceed with automated data fetching.")
    else:
        print("\n⚠️  Some tests failed. Consider manual CSV download instead.")
        print("   See docs/04_manual_data_download.md for instructions.")
    
    sys.exit(0 if (prices_ok and fundamentals_ok) else 1)
