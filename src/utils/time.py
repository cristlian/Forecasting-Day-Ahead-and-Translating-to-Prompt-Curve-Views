"""Time and timezone utilities."""

import logging
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


def localize_to_timezone(dt: datetime, timezone: str) -> datetime:
    """
    Localize a naive datetime to a timezone.
    
    Args:
        dt: Naive datetime
        timezone: Timezone string (e.g., 'Europe/Berlin')
    
    Returns:
        Timezone-aware datetime
    """
    tz = pytz.timezone(timezone)
    return tz.localize(dt)


def convert_timezone(dt: datetime, to_timezone: str) -> datetime:
    """
    Convert a timezone-aware datetime to another timezone.
    
    Args:
        dt: Timezone-aware datetime
        to_timezone: Target timezone string
    
    Returns:
        Datetime in target timezone
    """
    tz = pytz.timezone(to_timezone)
    return dt.astimezone(tz)


def create_hourly_range(
    start: datetime,
    end: datetime,
    timezone: str
) -> pd.DatetimeIndex:
    """
    Create an hourly datetime range in a specific timezone.
    
    Handles DST transitions correctly.
    
    Args:
        start: Start datetime
        end: End datetime
        timezone: Timezone string
    
    Returns:
        DatetimeIndex with hourly frequency
    """
    tz = pytz.timezone(timezone)
    
    # Create range in UTC first
    if start.tzinfo is None:
        start = tz.localize(start)
    if end.tzinfo is None:
        end = tz.localize(end)
    
    # Generate hourly range
    date_range = pd.date_range(start=start, end=end, freq='H')
    
    return date_range


def handle_dst_spring_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle spring DST transition (clock moves forward, hour disappears).
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with non-existent hour removed
    """
    # Identify non-existent times (DST spring forward)
    try:
        # Check if index can be converted to naive and back
        naive = df.index.tz_localize(None)
        _ = naive.tz_localize(df.index.tz, nonexistent='raise')
    except Exception:
        # Non-existent times detected, drop them
        logger.warning("Detected DST spring forward, dropping non-existent hour")
        df = df[~df.index.duplicated(keep=False)]
    
    return df


def handle_dst_fall_back(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle fall DST transition (clock moves back, hour repeats).
    
    Average the duplicate hour values.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with duplicate hour averaged
    """
    duplicates = df.index.duplicated(keep=False)
    
    if duplicates.any():
        logger.warning(f"Detected {duplicates.sum()} duplicate timestamps (DST fall back)")
        # Average duplicate entries
        df = df.groupby(level=0).mean()
    
    return df


def get_dst_transitions(year: int, timezone: str) -> List[datetime]:
    """
    Get DST transition dates for a given year and timezone.
    
    Args:
        year: Year
        timezone: Timezone string
    
    Returns:
        List of DST transition datetimes
    """
    tz = pytz.timezone(timezone)
    
    transitions = []
    
    # Check each day of the year
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                dt = datetime(year, month, day)
                if dt.replace(tzinfo=tz).dst() != (dt + timedelta(days=1)).replace(tzinfo=tz).dst():
                    transitions.append(dt)
            except ValueError:
                continue
    
    return transitions


def is_dst_active(dt: datetime) -> bool:
    """
    Check if DST is active at a given datetime.
    
    Args:
        dt: Timezone-aware datetime
    
    Returns:
        True if DST is active
    """
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")
    
    return dt.dst() != timedelta(0)
