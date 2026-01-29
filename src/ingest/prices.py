"""Day-ahead price data ingestion and normalization."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import pandas as pd

from .utils import (
    retry_on_failure,
    get_entsoe_api_key,
    get_area_code,
    to_utc,
    handle_dst,
    load_from_cache,
    save_to_cache,
    get_cache_path,
    ensure_hourly_frequency,
)

logger = logging.getLogger(__name__)


class PriceIngestionError(Exception):
    """Raised when price ingestion fails."""
    pass


def fetch_day_ahead_prices(
    market: str,
    start_date: datetime,
    end_date: datetime,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch day-ahead electricity prices from ENTSO-E or cache.
    
    Args:
        market: Market code (e.g., 'DE_LU')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        config: Configuration dictionary
        cache_dir: Directory for caching data
        force_refresh: If True, bypass cache
    
    Returns:
        DataFrame with columns: [timestamp (index), day_ahead_price]
        
    Raises:
        PriceIngestionError: If fetching fails and no cache available
    """
    logger.info(f"Fetching day-ahead prices for {market} from {start_date.date()} to {end_date.date()}")
    
    # Check cache first
    if cache_dir and not force_refresh:
        cache_path = get_cache_path(cache_dir, market, "prices", start_date, end_date)
        cached = load_from_cache(cache_path)
        if cached is not None:
            return cached
    
    # Try local file in data/raw (Manual Download)
    if cache_dir:
        try:
            raw_dir = cache_dir.parent / "raw"
            df = _fetch_from_local_file(market, start_date, end_date, raw_dir)
            if df is not None:
                logger.info(f"Successfully loaded {len(df)} price records from local file")
                # Normalize immediately as it's raw data
                df = normalize_prices(df, config)
                if cache_dir:
                    save_to_cache(df, cache_path)
                return df
        except Exception as e:
            logger.warning(f"Local file load failed: {e}")

    # Try ENTSO-E API
    api_key = get_entsoe_api_key()
    df = None
    
    if api_key:
        try:
            df = _fetch_from_entsoe(market, start_date, end_date, api_key, config)
            logger.info(f"Successfully fetched {len(df)} price records from ENTSO-E")
        except Exception as e:
            logger.warning(f"ENTSO-E fetch failed: {e}")
    
    # Fallback to SMARD if ENTSO-E fails
    if df is None:
        logger.info("Attempting SMARD fallback...")
        try:
            df = _fetch_from_smard(start_date, end_date)
            logger.info(f"Successfully fetched {len(df)} price records from SMARD")
        except Exception as e:
            logger.warning(f"SMARD fetch failed: {e}")
    
    # If both fail, check for any cached data
    if df is None:
        if cache_dir:
            # Try to find any existing cache file
            cache_pattern = cache_dir / market / "prices_*.parquet"
            existing_caches = list(cache_dir.glob(f"{market}/prices_*.parquet"))
            if existing_caches:
                latest_cache = max(existing_caches, key=lambda p: p.stat().st_mtime)
                logger.warning(f"Using stale cache: {latest_cache}")
                return pd.read_parquet(latest_cache)
        
        raise PriceIngestionError(
            f"Failed to fetch prices for {market}. "
            f"Set ENTSOE_API_KEY environment variable or provide cached data in {cache_dir}"
        )
    
    # Normalize and cache
    df = normalize_prices(df, config)
    
    if cache_dir:
        cache_path = get_cache_path(cache_dir, market, "prices", start_date, end_date)
        save_to_cache(df, cache_path)
    
    return df


@retry_on_failure(max_attempts=3, delay=2.0, backoff=2.0)
def _fetch_from_entsoe(
    market: str,
    start_date: datetime,
    end_date: datetime,
    api_key: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Fetch prices from ENTSO-E Transparency Platform."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        raise ImportError(
            "entsoe-py not installed. Install with: pip install entsoe-py"
        )
    
    client = EntsoePandasClient(api_key=api_key)
    area_code = get_area_code(market, config)
    
    # ENTSO-E requires timezone-aware dates
    tz = config.get("market", {}).get("timezone", "Europe/Berlin")
    start = pd.Timestamp(start_date, tz=tz)
    end = pd.Timestamp(end_date, tz=tz) + pd.Timedelta(days=1)  # Include end date
    
    # Fetch day-ahead prices
    prices = client.query_day_ahead_prices(area_code, start=start, end=end)
    
    # Convert Series to DataFrame
    if isinstance(prices, pd.Series):
        df = prices.to_frame(name="day_ahead_price")
    else:
        df = prices.rename(columns={prices.columns[0]: "day_ahead_price"})
    
    df.index.name = "timestamp"
    return df


def _fetch_from_local_file(
    market: str,
    start_date: datetime,
    end_date: datetime,
    raw_dir: Path,
) -> Optional[pd.DataFrame]:
    """Fetch prices from local CSV file in data/raw."""
    # Expected filename: prices_MARKET.csv (e.g., prices_DE_LU.csv)
    file_path = raw_dir / f"prices_{market}.csv"
    
    if not file_path.exists():
        logger.debug(f"Local file not found: {file_path}")
        return None
        
    logger.info(f"Loading prices from local file: {file_path}")
    
    # Try different separators and formats
    try:
        # Check first line to detect format
        with open(file_path, 'r') as f:
            first_line = f.readline()
            
        skip_rows = 0
        if "Day-ahead Prices" in first_line:
            skip_rows = 1  # Skip title row if present (ENTSO-E format)
        
        # Check if it's Energy-Charts format (timestamp,day_ahead_price)
        if 'timestamp' in first_line.lower() and 'day_ahead_price' in first_line.lower():
            # Energy-Charts format - already clean
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            df.index = pd.to_datetime(df.index, utc=True)
            logger.info(f"Detected Energy-Charts format, loaded {len(df)} records")
            return df
            
        df = pd.read_csv(file_path, sep=None, engine='python', skiprows=skip_rows)
    except Exception as e:
        logger.warning(f"Error reading CSV {file_path}: {e}")
        return None
        
    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '')

    # Identify columns
    price_col = None
    time_col = None
    
    for col in df.columns:
        c_lower = col.lower()
        if "price" in c_lower or "eur/mwh" in c_lower:
            price_col = col
        if "time" in c_lower or "mtu" in c_lower or "date" in c_lower:
            time_col = col
            
    if not price_col:
        logger.warning(f"Could not identify price column. Columns: {df.columns}")
        return None
        
    # Rename and Process
    df = df.rename(columns={price_col: "day_ahead_price"})
    
    # Handle timestamp
    if time_col:
        # ENTSO-E format: "dd.mm.yyyy HH:MM - dd.mm.yyyy HH:MM"
        # Take start time
        try:
            if df[time_col].dtype == object and df[time_col].str.contains(" - ").any():
                 df['timestamp'] = df[time_col].str.split(" - ").str[0]
                 # Generally ENTSO-E is DD.MM.YYYY HH:MM
                 df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
            else:
                 df['timestamp'] = pd.to_datetime(df[time_col], dayfirst=True)
        except Exception as e:
            logger.warning(f"Error parsing timestamps: {e}")
            return None
    else:
        # Fallback to first column
        try:
             df['timestamp'] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
        except:
             return None

    df = df.set_index('timestamp')
    df = df[['day_ahead_price']]
    
    # Ensure numeric
    df['day_ahead_price'] = pd.to_numeric(df['day_ahead_price'], errors='coerce')
    df = df.dropna()
    
    return df


def _fetch_from_smard(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch prices from SMARD.de (German energy data portal).
    No API key required.
    """
    import requests
    
    # SMARD API endpoint for day-ahead prices
    # Module 8004004 = Day-ahead spot market prices Germany/Luxembourg
    base_url = "https://www.smard.de/app/chart_data/8004004/DE/8004004_DE"
    
    all_data = []
    current = start_date
    
    while current <= end_date:
        # SMARD uses week-based files, get the Monday of the week
        week_start = current - timedelta(days=current.weekday())
        timestamp_ms = int(week_start.timestamp() * 1000)
        
        url = f"{base_url}_{timestamp_ms}.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse the series data
            series = data.get("series", [])
            for point in series:
                ts_ms, value = point
                if value is not None:
                    ts = pd.Timestamp(ts_ms, unit='ms', tz='Europe/Berlin')
                    all_data.append({"timestamp": ts, "day_ahead_price": value})
        except Exception as e:
            logger.debug(f"SMARD fetch for week {week_start.date()} failed: {e}")
        
        # Move to next week
        current = week_start + timedelta(days=7)
    
    if not all_data:
        raise PriceIngestionError("No data retrieved from SMARD")
    
    df = pd.DataFrame(all_data)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # Filter to requested range
    start_ts = pd.Timestamp(start_date, tz='Europe/Berlin')
    end_ts = pd.Timestamp(end_date, tz='Europe/Berlin') + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    
    return df


def normalize_prices(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize price data to standard schema.
    
    - Convert to UTC
    - Handle DST transitions
    - Ensure hourly frequency
    - Validate price ranges
    
    Args:
        df: Raw price dataframe
        config: Configuration dictionary
    
    Returns:
        Normalized dataframe with UTC index
    """
    logger.info("Normalizing price data")
    
    # Get timezone from config
    tz = config.get("market", {}).get("timezone", "Europe/Berlin")
    
    # Convert to UTC (internal standard)
    df = to_utc(df, source_tz=tz)
    
    # Handle DST transitions
    dst_policy = config.get("schema", {}).get("dst_policy", {
        "spring_forward": "drop",
        "fall_back": "average"
    })
    df = handle_dst(df, dst_policy)
    
    # Ensure hourly frequency (fill gaps with NaN for QA to catch)
    df = ensure_hourly_frequency(df)
    
    # Ensure column naming
    if "day_ahead_price" not in df.columns:
        # Try to find price column
        price_cols = [c for c in df.columns if 'price' in c.lower()]
        if price_cols:
            df = df.rename(columns={price_cols[0]: "day_ahead_price"})
    
    logger.info(f"Normalized {len(df)} hourly price records")
    return df
