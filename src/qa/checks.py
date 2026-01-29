"""Data quality checks for power market data."""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single QA check."""
    name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class QACheck:
    """Base class for QA checks."""
    
    def __init__(self, name: str):
        self.name = name
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Run the check. Returns CheckResult."""
        raise NotImplementedError


class CompletenessCheck(QACheck):
    """Check for missing data."""
    
    def __init__(self):
        super().__init__("Completeness Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check missing data against thresholds."""
        result = CheckResult(name=self.name, passed=True)
        
        thresholds = config.get("qa_thresholds", {}).get("completeness", {})
        schema = config.get("schema", {}).get("columns", {})
        
        critical_threshold = thresholds.get("max_missing_pct", {}).get("critical_columns", 1.0)
        optional_threshold = thresholds.get("max_missing_pct", {}).get("optional_columns", 10.0)
        
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
            
            result.metrics[f"{col}_missing_pct"] = missing_pct
            
            # Determine if column is required
            col_schema = schema.get(col, {})
            is_required = col_schema.get("required", False)
            threshold = critical_threshold if is_required else optional_threshold
            
            if missing_pct > threshold:
                msg = f"Column '{col}' has {missing_pct:.2f}% missing ({missing_count} rows), threshold: {threshold}%"
                result.errors.append(msg)
                result.passed = False
            elif missing_pct > 0:
                result.warnings.append(f"Column '{col}' has {missing_pct:.2f}% missing ({missing_count} rows)")
        
        return result


class DuplicateCheck(QACheck):
    """Check for duplicate timestamps."""
    
    def __init__(self):
        super().__init__("Duplicate Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check for duplicate timestamps."""
        result = CheckResult(name=self.name, passed=True)
        
        duplicates = df.index.duplicated()
        n_duplicates = duplicates.sum()
        
        threshold = config.get("qa_thresholds", {}).get("duplicates", {}).get("max_duplicate_timestamps", 0)
        result.metrics["duplicate_count"] = n_duplicates
        
        if n_duplicates > threshold:
            result.errors.append(f"Found {n_duplicates} duplicate timestamps (threshold: {threshold})")
            result.passed = False
            
            # Report some duplicate times
            dup_times = df.index[duplicates][:5]
            result.warnings.append(f"Sample duplicates: {[str(t) for t in dup_times]}")
        
        return result


class RangeCheck(QACheck):
    """Check if values are within expected ranges."""
    
    def __init__(self):
        super().__init__("Range Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check value ranges against schema."""
        result = CheckResult(name=self.name, passed=True)
        
        schema = config.get("schema", {}).get("columns", {})
        range_config = config.get("qa_thresholds", {}).get("ranges", {})
        fail_on_out_of_range = range_config.get("fail_on_out_of_range", True)
        warn_threshold_pct = range_config.get("warn_threshold_pct", 0.1)
        
        for col in df.columns:
            if col not in schema or "range" not in schema[col]:
                continue
            
            min_val, max_val = schema[col]["range"]
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            below_min = (col_data < min_val).sum()
            above_max = (col_data > max_val).sum()
            n_out = below_min + above_max
            pct_out = (n_out / len(col_data)) * 100
            
            result.metrics[f"{col}_out_of_range_pct"] = pct_out
            result.metrics[f"{col}_min"] = col_data.min()
            result.metrics[f"{col}_max"] = col_data.max()
            
            if n_out > 0:
                msg = f"Column '{col}': {n_out} values ({pct_out:.2f}%) out of range [{min_val}, {max_val}]"
                msg += f" (actual range: [{col_data.min():.2f}, {col_data.max():.2f}])"
                
                if fail_on_out_of_range and pct_out > warn_threshold_pct:
                    result.errors.append(msg)
                    result.passed = False
                else:
                    result.warnings.append(msg)
        
        return result


class TemporalCheck(QACheck):
    """Check temporal properties (gaps, frequency, continuity)."""
    
    def __init__(self):
        super().__init__("Temporal Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check for temporal issues."""
        result = CheckResult(name=self.name, passed=True)
        
        if len(df) < 2:
            result.warnings.append("Insufficient data for temporal checks")
            return result
        
        temporal_config = config.get("qa_thresholds", {}).get("temporal", {})
        max_gap_hours = temporal_config.get("max_gap_hours", 2)
        
        # Check for gaps
        time_diff = df.index.to_series().diff()
        expected_freq = pd.Timedelta(hours=1)
        
        gaps = time_diff[time_diff > expected_freq]
        
        result.metrics["max_gap_hours"] = time_diff.max().total_seconds() / 3600 if not time_diff.isna().all() else 0
        result.metrics["gap_count"] = len(gaps)
        
        if len(gaps) > 0:
            max_gap = time_diff.max()
            max_gap_hours_actual = max_gap.total_seconds() / 3600
            
            if max_gap_hours_actual > max_gap_hours:
                result.errors.append(f"Found gap of {max_gap_hours_actual:.1f} hours (max allowed: {max_gap_hours})")
                result.passed = False
            
            # Report gaps
            for ts, gap in gaps.items():
                gap_hours = gap.total_seconds() / 3600
                if gap_hours > 1:
                    result.warnings.append(f"Gap of {gap_hours:.1f}h ending at {ts}")
        
        # Check frequency consistency (allow some tolerance)
        irregular = (time_diff != expected_freq) & (time_diff.notna())
        n_irregular = irregular.sum()
        
        result.metrics["irregular_intervals"] = n_irregular
        
        # Exclude single gaps, just warn if many
        if n_irregular > len(df) * 0.01:  # More than 1% irregular
            result.warnings.append(f"Found {n_irregular} irregular time intervals (expected 1H)")
        
        # Check timestamp monotonicity
        if not df.index.is_monotonic_increasing:
            result.errors.append("Timestamps are not monotonically increasing")
            result.passed = False
        
        return result


class HourlyContinuityCheck(QACheck):
    """Check that each day has 24 hours (or 23/25 for DST)."""
    
    def __init__(self):
        super().__init__("Hourly Continuity Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check daily hour counts."""
        result = CheckResult(name=self.name, passed=True)
        
        if len(df) == 0:
            return result
        
        # Group by date and count hours
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        daily_counts = df_copy.groupby('date').size()
        
        # Most days should have 24 hours (23 or 25 for DST transitions)
        valid_counts = daily_counts.isin([23, 24, 25])
        invalid_days = daily_counts[~valid_counts]
        
        result.metrics["days_with_24h"] = (daily_counts == 24).sum()
        result.metrics["days_with_23h"] = (daily_counts == 23).sum()  # DST spring
        result.metrics["days_with_25h"] = (daily_counts == 25).sum()  # DST fall
        result.metrics["days_with_other"] = len(invalid_days)
        
        if len(invalid_days) > 0:
            for date, count in invalid_days.head(5).items():
                result.warnings.append(f"Date {date} has {count} hours (expected 23-25)")
            
            # Fail if too many invalid days
            if len(invalid_days) > len(daily_counts) * 0.05:  # More than 5%
                result.errors.append(f"{len(invalid_days)} days have irregular hour counts")
                result.passed = False
        
        return result


class AlignmentCheck(QACheck):
    """Check that price and fundamental data are aligned."""
    
    def __init__(self):
        super().__init__("Alignment Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check price-fundamental alignment."""
        result = CheckResult(name=self.name, passed=True)
        
        price_col = "day_ahead_price"
        fundamental_cols = ["forecast_load", "forecast_wind", "forecast_solar"]
        
        if price_col not in df.columns:
            result.errors.append(f"Price column '{price_col}' not found")
            result.passed = False
            return result
        
        # Check which fundamentals exist
        available_fundamentals = [c for c in fundamental_cols if c in df.columns]
        
        if not available_fundamentals:
            result.errors.append("No fundamental columns found")
            result.passed = False
            return result
        
        # Check alignment: rows where price exists but fundamentals don't (or vice versa)
        price_exists = ~df[price_col].isna()
        
        for col in available_fundamentals:
            fund_exists = ~df[col].isna()
            
            # Price exists but fundamental doesn't
            misaligned = (price_exists & ~fund_exists).sum()
            pct_misaligned = (misaligned / price_exists.sum()) * 100 if price_exists.sum() > 0 else 0
            
            result.metrics[f"{col}_misaligned_pct"] = pct_misaligned
            
            if pct_misaligned > 5:  # More than 5% misaligned
                result.warnings.append(f"{col}: {misaligned} rows ({pct_misaligned:.1f}%) missing where price exists")
        
        # Overall alignment score
        all_exist = price_exists.copy()
        for col in available_fundamentals:
            all_exist = all_exist & ~df[col].isna()
        
        aligned_pct = (all_exist.sum() / len(df)) * 100 if len(df) > 0 else 0
        result.metrics["fully_aligned_pct"] = aligned_pct
        
        if aligned_pct < 90:  # Less than 90% fully aligned
            result.warnings.append(f"Only {aligned_pct:.1f}% of rows have all data")
        
        return result


class OutlierCheck(QACheck):
    """Check for statistical outliers."""
    
    def __init__(self):
        super().__init__("Outlier Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> CheckResult:
        """Check for extreme outliers using z-score."""
        result = CheckResult(name=self.name, passed=True)
        
        outlier_config = config.get("qa_thresholds", {}).get("outliers", {})
        price_threshold = outlier_config.get("price_zscore_threshold", 10)
        gen_threshold = outlier_config.get("generation_zscore_threshold", 5)
        
        # Check price outliers
        if "day_ahead_price" in df.columns:
            price_data = df["day_ahead_price"].dropna()
            if len(price_data) > 10:
                z_scores = np.abs((price_data - price_data.mean()) / price_data.std())
                outliers = z_scores > price_threshold
                n_outliers = outliers.sum()
                
                result.metrics["price_outlier_count"] = n_outliers
                
                if n_outliers > 0:
                    result.warnings.append(f"Found {n_outliers} price outliers (|z| > {price_threshold})")
                    # Report extreme values
                    extreme = price_data[outliers].nlargest(3)
                    result.warnings.append(f"Extreme prices: {extreme.values.tolist()}")
        
        # Check generation outliers
        for col in ["forecast_load", "forecast_wind", "forecast_solar"]:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 10:
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outliers = z_scores > gen_threshold
                    n_outliers = outliers.sum()
                    
                    result.metrics[f"{col}_outlier_count"] = n_outliers
                    
                    if n_outliers > len(col_data) * 0.01:  # More than 1%
                        result.warnings.append(f"Found {n_outliers} outliers in {col}")
        
        # Outliers are warnings, not errors (don't fail the gate)
        return result


def run_all_checks(df: pd.DataFrame, config: Dict) -> List[CheckResult]:
    """
    Run all QA checks on a dataframe.
    
    Args:
        df: DataFrame to check (with datetime index)
        config: Configuration dictionary
    
    Returns:
        List of CheckResult objects
    """
    logger.info("Running QA checks")
    
    checks = [
        CompletenessCheck(),
        DuplicateCheck(),
        RangeCheck(),
        TemporalCheck(),
        HourlyContinuityCheck(),
        AlignmentCheck(),
        OutlierCheck(),
    ]
    
    results = []
    
    for check in checks:
        try:
            result = check.check(df, config)
            results.append(result)
            
            status = "PASSED" if result.passed else "FAILED"
            logger.info(f"{check.name}: {status}")
            
            for error in result.errors:
                logger.error(f"  [ERROR] {error}")
            for warning in result.warnings:
                logger.warning(f"  [WARN] {warning}")
                
        except Exception as e:
            logger.error(f"{check.name} raised exception: {e}")
            results.append(CheckResult(
                name=check.name,
                passed=False,
                errors=[f"Check failed with exception: {str(e)}"]
            ))
    
    return results
