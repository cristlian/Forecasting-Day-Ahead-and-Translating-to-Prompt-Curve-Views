"""Data quality checks."""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QACheck:
    """Base class for QA checks."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.warnings = []
        self.errors = []
    
    def check(self, df: pd.DataFrame, config: Dict) -> bool:
        """Run the check. Returns True if passed."""
        raise NotImplementedError


class CompletenessCheck(QACheck):
    """Check for missing data."""
    
    def __init__(self):
        super().__init__("Completeness Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> bool:
        """Check missing data against thresholds."""
        thresholds = config["qa_thresholds"]["completeness"]
        schema = config["schema"]["columns"]
        
        for col in df.columns:
            if col not in schema:
                continue
            
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            
            # Determine threshold
            is_required = schema[col].get("required", False)
            threshold = thresholds["max_missing_pct"]["critical_columns"] if is_required \
                       else thresholds["max_missing_pct"]["optional_columns"]
            
            if missing_pct > threshold:
                self.errors.append(
                    f"Column '{col}' has {missing_pct:.2f}% missing (threshold: {threshold}%)"
                )
            elif missing_pct > 0:
                self.warnings.append(
                    f"Column '{col}' has {missing_pct:.2f}% missing"
                )
        
        self.passed = len(self.errors) == 0
        return self.passed


class DuplicateCheck(QACheck):
    """Check for duplicate timestamps."""
    
    def __init__(self):
        super().__init__("Duplicate Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> bool:
        """Check for duplicate timestamps."""
        duplicates = df.index.duplicated()
        n_duplicates = duplicates.sum()
        
        threshold = config["qa_thresholds"]["duplicates"]["max_duplicate_timestamps"]
        
        if n_duplicates > threshold:
            self.errors.append(
                f"Found {n_duplicates} duplicate timestamps (threshold: {threshold})"
            )
        
        self.passed = len(self.errors) == 0
        return self.passed


class RangeCheck(QACheck):
    """Check if values are within expected ranges."""
    
    def __init__(self):
        super().__init__("Range Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> bool:
        """Check value ranges against schema."""
        schema = config["schema"]["columns"]
        
        for col in df.columns:
            if col not in schema or "range" not in schema[col]:
                continue
            
            min_val, max_val = schema[col]["range"]
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            n_out = out_of_range.sum()
            pct_out = (n_out / len(df)) * 100
            
            if n_out > 0:
                threshold = config["qa_thresholds"]["ranges"]["warn_threshold_pct"]
                
                msg = f"Column '{col}' has {n_out} values ({pct_out:.2f}%) out of range [{min_val}, {max_val}]"
                
                if config["qa_thresholds"]["ranges"]["fail_on_out_of_range"]:
                    self.errors.append(msg)
                elif pct_out > threshold:
                    self.warnings.append(msg)
        
        self.passed = len(self.errors) == 0
        return self.passed


class TemporalCheck(QACheck):
    """Check temporal properties (gaps, frequency)."""
    
    def __init__(self):
        super().__init__("Temporal Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> bool:
        """Check for temporal issues."""
        # Check for gaps
        time_diff = df.index.to_series().diff()
        max_gap = time_diff.max()
        expected_freq = pd.Timedelta(hours=1)
        
        max_gap_hours = config["qa_thresholds"]["temporal"]["max_gap_hours"]
        
        if max_gap > pd.Timedelta(hours=max_gap_hours):
            self.errors.append(
                f"Found gap of {max_gap} (max allowed: {max_gap_hours} hours)"
            )
        
        # Check frequency consistency
        irregular = (time_diff != expected_freq) & (time_diff.notna())
        if irregular.any():
            self.warnings.append(
                f"Found {irregular.sum()} irregular time intervals"
            )
        
        self.passed = len(self.errors) == 0
        return self.passed


class OutlierCheck(QACheck):
    """Check for statistical outliers."""
    
    def __init__(self):
        super().__init__("Outlier Check")
    
    def check(self, df: pd.DataFrame, config: Dict) -> bool:
        """Check for extreme outliers using z-score."""
        thresholds = config["qa_thresholds"]["outliers"]
        
        # Check price outliers
        if "day_ahead_price" in df.columns:
            z_scores = np.abs((df["day_ahead_price"] - df["day_ahead_price"].mean()) / 
                             df["day_ahead_price"].std())
            threshold = thresholds["price_zscore_threshold"]
            outliers = z_scores > threshold
            
            if outliers.any():
                self.warnings.append(
                    f"Found {outliers.sum()} price outliers (|z| > {threshold})"
                )
        
        self.passed = True  # Outliers are warnings, not errors
        return self.passed


def run_all_checks(df: pd.DataFrame, config: Dict) -> List[QACheck]:
    """
    Run all QA checks on a dataframe.
    
    Args:
        df: DataFrame to check
        config: Configuration dictionary
    
    Returns:
        List of QACheck results
    """
    logger.info("Running QA checks")
    
    checks = [
        CompletenessCheck(),
        DuplicateCheck(),
        RangeCheck(),
        TemporalCheck(),
        OutlierCheck(),
    ]
    
    for check in checks:
        check.check(df, config)
        
        if check.errors:
            logger.error(f"{check.name} FAILED:")
            for error in check.errors:
                logger.error(f"  - {error}")
        
        if check.warnings:
            logger.warning(f"{check.name} warnings:")
            for warning in check.warnings:
                logger.warning(f"  - {warning}")
        
        if check.passed:
            logger.info(f"{check.name} PASSED")
    
    return checks
