"""QA gate decision logic."""

import logging
from typing import List
from pathlib import Path
import pandas as pd

from .checks import CheckResult, run_all_checks
from .report import generate_qa_report

logger = logging.getLogger(__name__)


class QAGateFailure(Exception):
    """Raised when QA gate fails."""
    
    def __init__(self, message: str, failed_checks: List[str]):
        super().__init__(message)
        self.failed_checks = failed_checks


def evaluate_qa_gate(results: List[CheckResult]) -> bool:
    """
    Evaluate QA check results and decide whether to proceed.
    
    Args:
        results: List of CheckResult objects
    
    Returns:
        True if all checks passed
        
    Raises:
        QAGateFailure: If any critical check failed
    """
    failed = [r.name for r in results if not r.passed]
    
    if failed:
        logger.error(f"QA Gate FAILED. Failed checks: {', '.join(failed)}")
        raise QAGateFailure(
            f"QA gate failed: {len(failed)} checks did not pass",
            failed_checks=failed
        )
    
    logger.info("✓ QA Gate PASSED - All checks successful")
    return True


def run_qa_pipeline(
    df: pd.DataFrame,
    config: dict,
    run_id: str,
    output_dir: Path,
    fail_fast: bool = True,
) -> tuple[pd.DataFrame, List[CheckResult]]:
    """
    Run complete QA pipeline: checks, report generation, and gate evaluation.
    
    Args:
        df: DataFrame to validate (prices + fundamentals joined)
        config: Configuration dictionary
        run_id: Unique identifier for this run
        output_dir: Directory for QA reports
        fail_fast: If True, raise exception on failure
        
    Returns:
        Tuple of (cleaned DataFrame, list of check results)
        
    Raises:
        QAGateFailure: If fail_fast=True and checks fail
    """
    logger.info(f"Running QA pipeline for run_id: {run_id}")
    
    # Run all checks
    results = run_all_checks(df, config)
    
    # Generate reports
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc)
    generate_qa_report(results, output_dir, run_id, timestamp)
    
    # Evaluate gate
    all_passed = all(r.passed for r in results)
    
    if not all_passed:
        failed = [r.name for r in results if not r.passed]
        logger.error(f"QA checks failed: {failed}")
        
        if fail_fast:
            raise QAGateFailure(
                f"QA gate failed: {', '.join(failed)}",
                failed_checks=failed
            )
    
    # Clean the data (basic cleaning)
    df_clean = clean_dataset(df, config)
    
    return df_clean, results


def clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean dataset after QA checks pass.
    
    Applies:
    - Forward fill for small gaps (up to 2 hours)
    - Drop remaining NaN rows
    - Ensure sorted index
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning dataset")
    
    df_clean = df.copy()
    
    # Ensure sorted
    df_clean = df_clean.sort_index()
    
    # Get max gap hours from config
    max_gap = config.get("qa_thresholds", {}).get("temporal", {}).get("max_gap_hours", 2)
    
    # Forward fill small gaps (up to max_gap hours)
    df_clean = df_clean.ffill(limit=max_gap)
    
    # Log cleaning stats
    original_len = len(df)
    remaining_na = df_clean.isna().any(axis=1).sum()
    
    # Only drop rows if they have missing required columns
    required_cols = ["day_ahead_price"]  # At minimum need the target
    if all(col in df_clean.columns for col in required_cols):
        df_clean = df_clean.dropna(subset=required_cols)
    
    final_len = len(df_clean)
    dropped = original_len - final_len
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing required data ({dropped/original_len*100:.2f}%)")
    
    logger.info(f"Clean dataset: {final_len} rows, {len(df_clean.columns)} columns")
    
    return df_clean


def get_qa_summary(results: List[CheckResult]) -> dict:
    """
    Get a summary of QA results for reporting.
    
    Args:
        results: List of CheckResult objects
        
    Returns:
        Summary dictionary
    """
    return {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "warnings": sum(len(r.warnings) for r in results),
        "errors": sum(len(r.errors) for r in results),
        "all_passed": all(r.passed for r in results),
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "error_count": len(r.errors),
                "warning_count": len(r.warnings),
            }
            for r in results
        ]
    }
