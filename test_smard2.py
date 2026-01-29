"""Test alternative data sources for German power data."""
import requests
import json
from datetime import datetime

print("=" * 60)
print("Testing Alternative Data Sources (No API Key Required)")
print("=" * 60)

# ============================================================
# 1. Energy-Charts API (Fraunhofer ISE) - Usually works!
# ============================================================
print("\n1. Testing Energy-Charts API (Fraunhofer ISE)...")

base_url = "https://api.energy-charts.info"

# Test Day-Ahead Prices
print("\n   a) Day-Ahead Prices...")
try:
    r = requests.get(
        f"{base_url}/price",
        params={'bzn': 'DE-LU', 'start': '2024-12-01', 'end': '2024-12-07'},
        timeout=30
    )
    print(f"      Status: {r.status_code}")
    if r.ok:
        data = r.json()
        print(f"      ✅ SUCCESS! Keys: {list(data.keys())}")
        if 'price' in data:
            prices = data['price']
            print(f"      Got {len(prices)} price values")
            print(f"      Sample: {prices[:5]}")
except Exception as e:
    print(f"      Error: {e}")

# Test Load data
print("\n   b) Load Forecast...")
try:
    r = requests.get(
        f"{base_url}/total_power",
        params={'country': 'de', 'start': '2024-12-01', 'end': '2024-12-07'},
        timeout=30
    )
    print(f"      Status: {r.status_code}")
    if r.ok:
        data = r.json()
        print(f"      ✅ SUCCESS! Keys: {list(data.keys())}")
except Exception as e:
    print(f"      Error: {e}")

# Test Wind/Solar
print("\n   c) Power Generation (includes Wind/Solar)...")
try:
    r = requests.get(
        f"{base_url}/public_power",
        params={'country': 'de', 'start': '2024-12-01', 'end': '2024-12-07'},
        timeout=30
    )
    print(f"      Status: {r.status_code}")
    if r.ok:
        data = r.json()
        print(f"      ✅ SUCCESS! Keys: {list(data.keys())[:10]}...")
except Exception as e:
    print(f"      Error: {e}")

# ============================================================
# 2. OPSD (Open Power System Data) - Static datasets
# ============================================================
print("\n2. Testing OPSD (Open Power System Data)...")
try:
    # Time series dataset
    url = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
    r = requests.head(url, timeout=10)
    print(f"   Status: {r.status_code}")
    if r.ok:
        size = int(r.headers.get('Content-Length', 0)) / 1024 / 1024
        print(f"   ✅ Available! Size: {size:.1f} MB")
        print(f"   URL: {url}")
except Exception as e:
    print(f"   Error: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
If Energy-Charts API works, we can use it as the primary data source!
- No API key required
- Covers DE-LU bidding zone
- Has prices, load, wind, solar data
- Free and reliable
""")
