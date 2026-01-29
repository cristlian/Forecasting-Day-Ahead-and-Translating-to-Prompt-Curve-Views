"""Fundamental data ingestion (load forecasts, generation forecasts)."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import pandas as pd

from .utils import (
    retry_on_failure,
    to_utc,
    handle_dst,
    load_from_cache,
    save_to_cache,
    get_cache_path,
    ensure_hourly_frequency,
)

logger = logging.getLogger(__name__)


class FundamentalsIngestionError(Exception):
    """Raised when fundamentals ingestion fails."""
    pass


def fetch_fundamentals(
    market: str,
    start_date: datetime,
    end_date: datetime,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch all fundamental drivers (load forecast, wind forecast, solar forecast).
    
    IMPORTANT: We fetch DAY-AHEAD FORECASTS, not actuals, to avoid look-ahead bias.
    
    Args:
        market: Market code (e.g., 'DE_LU')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        config: Configuration dictionary
        cache_dir: Directory for caching data
        force_refresh: If True, bypass cache
    
    Returns:
        DataFrame with columns: [timestamp (index), forecast_load, forecast_wind, forecast_solar]
    """
    logger.info(f"Fetching fundamentals for {market} from {start_date.date()} to {end_date.date()}")
    
    # Check cache first
    if cache_dir and not force_refresh:
        cache_path = get_cache_path(cache_dir, market, "fundamentals", start_date, end_date)
        cached = load_from_cache(cache_path)
        if cached is not None:
            tz = config.get("market", {}).get("timezone", "Europe/Berlin")
            start_ts = pd.Timestamp(start_date, tz=tz).tz_convert("UTC")
            end_ts = (pd.Timestamp(end_date, tz=tz) + pd.Timedelta(days=1)).tz_convert("UTC")
            cached = cached[(cached.index >= start_ts) & (cached.index < end_ts)]
            return cached
    
    # Try local files in data/raw (Manual Download)
    if cache_dir:
        try:
            raw_dir = cache_dir.parent / "raw"
            df = _fetch_fundamentals_local(market, start_date, end_date, raw_dir)
            if df is not None:
                logger.info(f"Successfully loaded {len(df)} fundamental records from local files")
                df = normalize_fundamentals(df, config)
                if cache_dir:
                    save_to_cache(df, cache_path)
                return df
        except Exception as e:
            logger.warning(f"Local fundamentals load failed: {e}")

    # Fallback to SMARD if local files are not available
    df = None
    logger.info("Attempting SMARD fallback for fundamentals...")
    try:
        df = _fetch_all_from_smard(start_date, end_date)
        logger.info(f"Successfully fetched {len(df)} fundamental records from SMARD")
    except Exception as e:
        logger.warning(f"SMARD fundamentals fetch failed: {e}")
    
    # If both fail, check for any cached data
    if df is None:
        if cache_dir:
            existing_caches = list(cache_dir.glob(f"{market}/fundamentals_*.parquet"))
            if existing_caches:
                latest_cache = max(existing_caches, key=lambda p: p.stat().st_mtime)
                logger.warning(f"Using stale cache: {latest_cache}")
                return pd.read_parquet(latest_cache)
        
        raise FundamentalsIngestionError(
            f"Failed to fetch fundamentals for {market}. "
            f"Provide local raw data in {cache_dir.parent / 'raw'} or cached data."
        )
    
    # Normalize and cache
    df = normalize_fundamentals(df, config)
    
    if cache_dir:
        cache_path = get_cache_path(cache_dir, market, "fundamentals", start_date, end_date)
        save_to_cache(df, cache_path)
    
    return df


def _fetch_fundamentals_local(
    market: str,
    start_date: datetime,
    end_date: datetime,
    raw_dir: Path,
) -> Optional[pd.DataFrame]:
    """Fetch fundamentals from local CSV files in data/raw."""
    # Files: load_MARKET.csv, gen_forecast_MARKET.csv
    dfs = []
    
    # Helper to load Energy-Charts format (already clean with timestamp index)
    def load_energy_charts_csv(path, expected_cols):
        if not path.exists(): return None
        try:
            with open(path, 'r') as f:
                first_line = f.readline().lower()
            # Energy-Charts format: timestamp index with named columns
            if 'timestamp' in first_line and any(col in first_line for col in expected_cols):
                df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
                df.index = pd.to_datetime(df.index, utc=True)
                logger.info(f"Loaded Energy-Charts format from {path.name}: {len(df)} records")
                return df
        except Exception as e:
            logger.debug(f"Not Energy-Charts format: {e}")
        return None
    
    # Try Energy-Charts format first (already properly formatted)
    # 1. Load Forecast (Energy-Charts format)
    load_path = raw_dir / f"load_{market}.csv"
    df_ec_load = load_energy_charts_csv(load_path, ['forecast_load'])
    if df_ec_load is not None and 'forecast_load' in df_ec_load.columns:
        dfs.append(df_ec_load[['forecast_load']])
    
    # 2. Generation Forecast (Energy-Charts format)  
    gen_path = raw_dir / f"gen_forecast_{market}.csv"
    df_ec_gen = load_energy_charts_csv(gen_path, ['forecast_wind', 'forecast_solar'])
    if df_ec_gen is not None:
        if 'forecast_wind' in df_ec_gen.columns:
            dfs.append(df_ec_gen[['forecast_wind']])
        if 'forecast_solar' in df_ec_gen.columns:
            dfs.append(df_ec_gen[['forecast_solar']])
    
    # If Energy-Charts format worked, return early
    if dfs:
        df = pd.concat(dfs, axis=1)
        df = df.sort_index()
        mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & (df.index <= pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1))
        df = df[mask]
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    
    if not dfs:
        return None
        
    df = pd.concat(dfs, axis=1)
    
    # Filter
    df = df.sort_index()
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1))
    df = df[mask]
    
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df


def _fetch_all_from_smard(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch fundamentals from SMARD.de.
    
    SMARD module IDs:
    - 4067: Prognostizierte Erzeugung Wind Onshore
    - 4068: Prognostizierte Erzeugung Wind Offshore  
    - 4069: Prognostizierte Erzeugung Photovoltaik
    - 4065: Prognostizierte Netzlast (Grid Load Forecast)
    """
    import requests
    
    modules = {
        "forecast_load": "4065",
        "forecast_wind_onshore": "4067",
        "forecast_wind_offshore": "4068",
        "forecast_solar": "4069",
    }
    
    all_series = {}
    
    for name, module_id in modules.items():
        series_data = []
        current = start_date
        
        while current <= end_date:
            week_start = current - timedelta(days=current.weekday())
            timestamp_ms = int(week_start.timestamp() * 1000)
            
            url = f"https://www.smard.de/app/chart_data/{module_id}/DE/{module_id}_DE_{timestamp_ms}.json"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                series = data.get("series", [])
                for point in series:
                    ts_ms, value = point
                    if value is not None:
                        ts = pd.Timestamp(ts_ms, unit='ms', tz='Europe/Berlin')
                        series_data.append({"timestamp": ts, name: value})
            except Exception as e:
                logger.debug(f"SMARD {name} fetch for week {week_start.date()} failed: {e}")
            
            current = week_start + timedelta(days=7)
        
        if series_data:
            df = pd.DataFrame(series_data)
            df.set_index("timestamp", inplace=True)
            all_series[name] = df[name]
    
    if not all_series:
        raise FundamentalsIngestionError("No fundamental data retrieved from SMARD")
    
    # Combine all series
    result = pd.DataFrame(all_series)
    result.sort_index(inplace=True)
    
    # Combine wind onshore + offshore
    if "forecast_wind_onshore" in result.columns and "forecast_wind_offshore" in result.columns:
        result["forecast_wind"] = result["forecast_wind_onshore"].fillna(0) + result["forecast_wind_offshore"].fillna(0)
        result.drop(columns=["forecast_wind_onshore", "forecast_wind_offshore"], inplace=True)
    elif "forecast_wind_onshore" in result.columns:
        result["forecast_wind"] = result["forecast_wind_onshore"]
        result.drop(columns=["forecast_wind_onshore"], inplace=True)
    
    # Filter to requested range
    start_ts = pd.Timestamp(start_date, tz='Europe/Berlin')
    end_ts = pd.Timestamp(end_date, tz='Europe/Berlin') + pd.Timedelta(days=1)
    result = result[(result.index >= start_ts) & (result.index < end_ts)]
    
    result.index.name = "timestamp"
    return result


def normalize_fundamentals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize fundamental data to standard schema.
    
    Args:
        df: Raw fundamentals dataframe
        config: Configuration dictionary
    
    Returns:
        Normalized dataframe with UTC index
    """
    logger.info("Normalizing fundamentals data")
    
    tz = config.get("market", {}).get("timezone", "Europe/Berlin")
    
    # Convert to UTC
    df = to_utc(df, source_tz=tz)
    
    # Handle DST
    dst_policy = config.get("schema", {}).get("dst_policy", {
        "spring_forward": "drop",
        "fall_back": "average"
    })
    df = handle_dst(df, dst_policy)
    
    # Ensure hourly frequency
    df = ensure_hourly_frequency(df)
    
    # Ensure expected columns exist
    expected_cols = ["forecast_load", "forecast_wind", "forecast_solar"]
    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Missing expected column: {col}")
            df[col] = float('nan')

    # Clamp invalid load values
    if "forecast_load" in df.columns:
        df.loc[df["forecast_load"] < 0, "forecast_load"] = float('nan')
    
    # Keep only expected columns
    df = df[[c for c in expected_cols if c in df.columns]]
    
    logger.info(f"Normalized {len(df)} hourly fundamental records")
    return df


def fetch_load_forecast(
    market: str,
    start_date: datetime,
    end_date: datetime,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience function to fetch only load forecast."""
    df = fetch_fundamentals(market, start_date, end_date, config, cache_dir)
    return df[["forecast_load"]] if "forecast_load" in df.columns else df


def fetch_wind_forecast(
    market: str,
    start_date: datetime,
    end_date: datetime,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience function to fetch only wind forecast."""
    df = fetch_fundamentals(market, start_date, end_date, config, cache_dir)
    return df[["forecast_wind"]] if "forecast_wind" in df.columns else df


def fetch_solar_forecast(
    market: str,
    start_date: datetime,
    end_date: datetime,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience function to fetch only solar forecast."""
    df = fetch_fundamentals(market, start_date, end_date, config, cache_dir)
    return df[["forecast_solar"]] if "forecast_solar" in df.columns else df


