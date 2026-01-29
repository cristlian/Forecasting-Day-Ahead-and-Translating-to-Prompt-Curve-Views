#!/bin/bash
# Download a small sample dataset for reviewers

set -e

echo "=== Downloading Sample Data ==="
echo ""

# This is a placeholder script
# In a real implementation, you would:
# 1. Download sample data from a public source
# 2. Or generate synthetic data
# 3. Save to data/raw/

SAMPLE_DIR="data/raw"
mkdir -p $SAMPLE_DIR

echo "Generating synthetic sample data..."

# Use Python to generate sample data
python << EOF
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 30 days of hourly data
start_date = datetime(2024, 1, 1)
dates = pd.date_range(start_date, periods=30*24, freq='H', tz='Europe/Berlin')

# Synthetic data
np.random.seed(42)

df = pd.DataFrame({
    'timestamp': dates,
    'day_ahead_price': 50 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.randn(len(dates)) * 5,
    'actual_load': 50000 + 10000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.randn(len(dates)) * 1000,
    'wind_generation': 15000 + 5000 * np.random.rand(len(dates)),
    'solar_generation': np.maximum(0, 8000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)),
})

# Ensure solar is 0 at night
df.loc[df['timestamp'].dt.hour < 6, 'solar_generation'] = 0
df.loc[df['timestamp'].dt.hour > 20, 'solar_generation'] = 0

# Save
df.to_parquet('$SAMPLE_DIR/sample_data.parquet', index=False)
print(f"Generated {len(df)} rows of sample data")
print(f"Saved to $SAMPLE_DIR/sample_data.parquet")

EOF

echo ""
echo "✓ Sample data generated!"
echo "Location: $SAMPLE_DIR/sample_data.parquet"
echo ""
echo "Note: This is synthetic data for testing purposes only."
