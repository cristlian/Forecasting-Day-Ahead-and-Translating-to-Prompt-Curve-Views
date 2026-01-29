"""Quick test of Gemini API and data files."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("1. Checking raw data files...")
print("=" * 50)

raw = Path('data/raw')
files = {
    'prices_DE_LU.csv': 'day_ahead_price',
    'load_DE_LU.csv': 'forecast_load', 
    'gen_forecast_DE_LU.csv': 'forecast_wind'
}

all_ok = True
for f, col in files.items():
    path = raw / f
    if path.exists() and path.stat().st_size > 1000:
        import pandas as pd
        df = pd.read_csv(path)
        print(f"  ✅ {f}: {len(df)} rows, has '{col}': {col in df.columns or 'timestamp' in df.columns}")
    else:
        print(f"  ❌ {f}: MISSING or empty!")
        all_ok = False

print("\n" + "=" * 50)
print("2. Checking Gemini API...")
print("=" * 50)

key = os.getenv('GEMINI_API_KEY')
if key:
    print(f"  ✅ GEMINI_API_KEY found: {key[:15]}...")
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Reply with just: OK'
        )
        print(f"  ✅ Gemini API works! Response: {response.text.strip()}")
    except Exception as e:
        print(f"  ❌ Gemini API error: {e}")
else:
    print("  ❌ GEMINI_API_KEY not found in .env")

print("\n" + "=" * 50)
print("3. Summary")
print("=" * 50)
print(f"  Data files: {'✅ Ready' if all_ok else '❌ Need download'}")
print(f"  Gemini API: {'✅ Ready' if key else '❌ Need key'}")
