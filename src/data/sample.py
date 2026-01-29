"""Sample data generation for offline testing and demos.

Generates realistic synthetic power market data for the DE-LU market.
This allows running the full pipeline without API keys.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


def generate_sample_features(
    start_date: datetime = None,
    end_date: datetime = None,
    n_days: int = 60,
    market: str = "DE_LU",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic sample feature data for power price forecasting.
    
    This creates synthetic data with realistic patterns:
    - Daily and weekly seasonality in prices
    - Correlated fundamentals (load, wind, solar)
    - Proper feature engineering already applied
    
    Args:
        start_date: Start date (defaults to 60 days ago)
        end_date: End date (defaults to today)
        n_days: Number of days if dates not specified
        market: Market code
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with features ready for model training
    """
    np.random.seed(seed)
    
    if start_date is None:
        end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=n_days)
    elif end_date is None:
        end_date = start_date + timedelta(days=n_days)
    
    # Generate hourly timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')[:-1]
    n_hours = len(timestamps)
    
    logger.info(f"Generating sample data: {start_date.date()} to {end_date.date()} ({n_hours} hours)")
    
    # Hour and day features
    hours = np.array([t.hour for t in timestamps])
    days_of_week = np.array([t.dayofweek for t in timestamps])
    months = np.array([t.month for t in timestamps])
    
    # Base price pattern (daily cycle + weekly pattern)
    base_price = 50.0
    
    # Daily pattern: peaks at 8-9 AM and 6-7 PM
    hour_effect = -15 * np.cos(2 * np.pi * hours / 24) + 5 * np.sin(4 * np.pi * hours / 24)
    
    # Weekly pattern: lower on weekends
    weekend_effect = np.where(days_of_week >= 5, -10, 0)
    
    # Seasonal effect: higher in winter
    seasonal_effect = 10 * np.cos(2 * np.pi * (months - 1) / 12)
    
    # Generate fundamentals
    # Load: follows daily pattern, higher on weekdays
    base_load = 55000
    load_daily = 10000 * np.sin(2 * np.pi * (hours - 6) / 24)
    load_weekly = np.where(days_of_week >= 5, -5000, 0)
    forecast_load = base_load + load_daily + load_weekly + np.random.normal(0, 1000, n_hours)
    
    # Wind: more variable, somewhat higher at night
    base_wind = 15000
    wind_daily = 2000 * np.cos(2 * np.pi * hours / 24)
    forecast_wind = np.maximum(0, base_wind + wind_daily + np.random.normal(0, 5000, n_hours))
    
    # Solar: follows sun (zero at night, peak at noon)
    solar_factor = np.maximum(0, np.sin(np.pi * (hours - 6) / 12)) ** 2
    solar_factor = np.where((hours >= 6) & (hours <= 18), solar_factor, 0)
    base_solar = 10000
    forecast_solar = base_solar * solar_factor + np.random.normal(0, 1000, n_hours)
    forecast_solar = np.maximum(0, forecast_solar)
    
    # Residual load = Load - Wind - Solar
    forecast_residual_load = forecast_load - forecast_wind - forecast_solar
    
    # Price depends on residual load + randomness
    price_from_residual = 0.001 * (forecast_residual_load - 30000)
    noise = np.random.normal(0, 8, n_hours)
    
    day_ahead_price = (
        base_price + hour_effect + weekend_effect + seasonal_effect + 
        price_from_residual + noise
    )
    
    # Ensure positive prices (can have negative in reality, but keep simple)
    day_ahead_price = np.maximum(day_ahead_price, -20)
    
    # Create base DataFrame
    df = pd.DataFrame({
        'day_ahead_price': day_ahead_price,
        'forecast_load': forecast_load,
        'forecast_wind': forecast_wind,
        'forecast_solar': forecast_solar,
        'forecast_residual_load': forecast_residual_load,
    }, index=timestamps)
    df.index.name = 'timestamp'
    
    # Add calendar features
    df['hour'] = hours
    df['day_of_week'] = days_of_week
    df['month'] = months
    df['is_weekend'] = (days_of_week >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['dow_sin'] = np.sin(2 * np.pi * days_of_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * days_of_week / 7)
    
    # Add lag features (minimum 24h for no leakage)
    for lag in [24, 48, 168]:  # 1 day, 2 days, 1 week
        df[f'lag_{lag}h_day_ahead_price'] = df['day_ahead_price'].shift(lag)
        df[f'lag_{lag}h_forecast_load'] = df['forecast_load'].shift(lag)
    
    # Add rolling features (looking backward only)
    for window in [24, 168]:
        df[f'roll_{window}h_mean_day_ahead_price'] = (
            df['day_ahead_price'].shift(24).rolling(window, min_periods=12).mean()
        )
        df[f'roll_{window}h_std_day_ahead_price'] = (
            df['day_ahead_price'].shift(24).rolling(window, min_periods=12).std()
        )
    
    # Add interaction features
    df['hour_x_dow'] = df['hour'] * df['day_of_week']
    df['wind_x_solar'] = df['forecast_wind'] * df['forecast_solar'] / 1e6
    
    logger.info(f"Generated {len(df)} rows with {len(df.columns)} columns")
    
    return df


def save_sample_features(
    output_dir: Path,
    start_date: datetime = None,
    end_date: datetime = None,
    n_days: int = 60,
    market: str = "DE_LU",
) -> Path:
    """
    Generate and save sample features to disk.
    
    Args:
        output_dir: Directory to save features
        start_date: Start date
        end_date: End date  
        n_days: Number of days if dates not specified
        market: Market code
    
    Returns:
        Path to saved features file
    """
    df = generate_sample_features(
        start_date=start_date,
        end_date=end_date,
        n_days=n_days,
        market=market,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "sample_features.parquet"
    df.to_parquet(output_path)
    
    logger.info(f"Saved sample features to {output_path}")
    
    return output_path


def get_sample_features_path(root_dir: Path = None) -> Path:
    """Get path to sample features file."""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent
    return root_dir / "data" / "sample" / "sample_features.parquet"


def load_sample_features(root_dir: Path = None) -> pd.DataFrame:
    """
    Load sample features, generating if not present.
    
    Args:
        root_dir: Project root directory
    
    Returns:
        DataFrame with sample features
    """
    sample_path = get_sample_features_path(root_dir)
    
    if sample_path.exists():
        logger.info(f"Loading sample features from {sample_path}")
        return pd.read_parquet(sample_path)
    
    # Generate if not present
    logger.info("Sample features not found, generating...")
    save_sample_features(sample_path.parent)
    return pd.read_parquet(sample_path)
