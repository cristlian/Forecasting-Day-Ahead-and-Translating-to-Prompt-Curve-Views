"""Utility functions for data ingestion."""

import time
import logging
import hashlib
from datetime import datetime, timezone
from typing import Callable, Any, Optional
from functools import wraps
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each attempt
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def get_cache_path(cache_dir: Path, market: str, data_type: str, 
                   start_date: datetime, end_date: datetime) -> Path:
    """Generate a deterministic cache file path."""
    # Create hash of parameters for unique filename
    key = f"{market}_{data_type}_{start_date.date()}_{end_date.date()}"
    filename = f"{data_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    return cache_dir / market / filename


def load_from_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load data from cache if exists."""
    if cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    return None


def save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save data to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info(f"Cached data to: {cache_path}")


def to_utc(df: pd.DataFrame, source_tz: str = "Europe/Berlin") -> pd.DataFrame:
    """
    Convert DataFrame index to UTC.
    
    Args:
        df: DataFrame with datetime index
        source_tz: Source timezone if index is naive
        
    Returns:
        DataFrame with UTC-aware index
    """
    if df.index.tz is None:
        df.index = df.index.tz_localize(source_tz, ambiguous='infer', nonexistent='shift_forward')
    df.index = df.index.tz_convert("UTC")
    return df


def handle_dst(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """
    Handle DST transitions according to policy.
    
    Args:
        df: DataFrame with datetime index
        policy: Dict with 'spring_forward' and 'fall_back' keys
        
    Returns:
        DataFrame with DST handled
    """
    # Check for duplicates (fall back - repeated hour)
    duplicates = df.index.duplicated(keep=False)
    if duplicates.any():
        n_dups = duplicates.sum() // 2
        logger.info(f"Handling {n_dups} duplicate hours (DST fall back)")
        if policy.get("fall_back") == "average":
            df = df.groupby(level=0).mean()
        else:  # keep first by default
            df = df[~df.index.duplicated(keep='first')]
    
    return df


def generate_run_id(market: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a unique run ID."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    return f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{market}"


def ensure_hourly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has continuous hourly frequency.
    
    Fills missing hours with NaN (to be handled by QA).
    """
    if len(df) == 0:
        return df
    
    # Create complete hourly range
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h',
        tz=df.index.tz
    )
    
    # Reindex to fill gaps
    df = df.reindex(full_range)
    df.index.name = 'timestamp'
    
    return df


