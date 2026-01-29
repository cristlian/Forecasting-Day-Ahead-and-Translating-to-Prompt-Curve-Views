"""Utility functions for data ingestion."""

import time
import logging
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


def cache_data(cache_dir: str, filename: str):
    """
    Decorator to cache fetched data to disk.
    
    Args:
        cache_dir: Directory to store cached data
        filename: Filename for cached data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import pandas as pd
            from pathlib import Path
            
            cache_path = Path(cache_dir) / filename
            
            # Check if cache exists
            if cache_path.exists():
                logger.info(f"Loading from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            
            # Fetch and cache
            logger.info(f"Cache miss, fetching data")
            data = func(*args, **kwargs)
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(cache_path)
            logger.info(f"Cached data to: {cache_path}")
            
            return data
        return wrapper
    return decorator


def parse_entsoe_response(response: Any) -> Any:
    """
    Parse ENTSO-E API response and handle common issues.
    
    Args:
        response: Raw API response
    
    Returns:
        Parsed data
    """
    # TODO: Implement ENTSO-E response parsing
    # Handle XML/JSON formats
    # Handle error responses
    # Handle missing data indicators
    raise NotImplementedError("ENTSO-E response parsing not yet implemented")


def convert_timezone(df, from_tz: str, to_tz: str):
    """
    Convert dataframe index from one timezone to another.
    
    Args:
        df: DataFrame with datetime index
        from_tz: Source timezone
        to_tz: Target timezone
    
    Returns:
        DataFrame with converted timezone
    """
    if df.index.tz is None:
        df.index = df.index.tz_localize(from_tz)
    else:
        df.index = df.index.tz_convert(to_tz)
    
    return df
