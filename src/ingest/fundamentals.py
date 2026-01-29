"""Fundamental data ingestion (load, generation, weather)."""

import logging
from datetime import datetime
from typing import Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_load(
    market: str,
    start_date: datetime,
    end_date: datetime,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch actual load (demand) data.
    
    Args:
        market: Market code
        start_date: Start date
        end_date: End date
        api_key: API key (if required)
    
    Returns:
        DataFrame with [timestamp, actual_load]
    """
    logger.info(f"Fetching load data for {market}")
    
    # TODO: Implement ENTSO-E actual load query
    raise NotImplementedError("Load fetching not yet implemented")


def fetch_generation(
    market: str,
    start_date: datetime,
    end_date: datetime,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch generation data by type (wind, solar, etc.).
    
    Args:
        market: Market code
        start_date: Start date
        end_date: End date
        api_key: API key (if required)
    
    Returns:
        DataFrame with [timestamp, wind_generation, solar_generation, ...]
    """
    logger.info(f"Fetching generation data for {market}")
    
    # TODO: Implement ENTSO-E generation query
    # Separate queries for wind onshore, wind offshore, solar
    # Aggregate onshore + offshore wind
    
    raise NotImplementedError("Generation fetching not yet implemented")


def fetch_weather(
    location: Dict[str, float],
    start_date: datetime,
    end_date: datetime,
    variables: list = ["temperature_2m"]
) -> pd.DataFrame:
    """
    Fetch weather data (temperature, wind speed, etc.).
    
    Args:
        location: Dict with 'latitude' and 'longitude'
        start_date: Start date
        end_date: End date
        variables: List of weather variables to fetch
    
    Returns:
        DataFrame with [timestamp, temperature, ...]
    """
    logger.info(f"Fetching weather data for location {location}")
    
    # TODO: Implement Open-Meteo API call
    # Example:
    # import openmeteo_requests
    # client = openmeteo_requests.Client()
    # params = {
    #     "latitude": location["latitude"],
    #     "longitude": location["longitude"],
    #     "start_date": start_date.strftime("%Y-%m-%d"),
    #     "end_date": end_date.strftime("%Y-%m-%d"),
    #     "hourly": variables
    # }
    # responses = client.weather_api("...", params=params)
    
    raise NotImplementedError("Weather fetching not yet implemented")


def normalize_fundamentals(
    df: pd.DataFrame,
    timezone: str,
    schema: Dict
) -> pd.DataFrame:
    """
    Normalize fundamental data to standard schema.
    
    Args:
        df: Raw fundamentals dataframe
        timezone: Target timezone
        schema: Schema configuration dict
    
    Returns:
        Normalized dataframe
    """
    logger.info("Normalizing fundamental data")
    
    # Convert to timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)
    
    # Ensure hourly frequency
    df = df.resample('H').mean()
    
    # Validate column names and units
    for col in df.columns:
        if col not in schema["columns"]:
            logger.warning(f"Column {col} not in schema, dropping")
            df = df.drop(columns=[col])
    
    return df
