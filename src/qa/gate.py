"""QA gate decision logic."""

import logging
from typing import List

logger = logging.getLogger(__name__)


def evaluate_qa_gate(checks: List) -> bool:
    """
    Evaluate QA checks and decide whether to proceed.
    
    Args:
        checks: List of QACheck results
    
    Returns:
        True if all checks passed, False otherwise
    
    Raises:
        QAGateFailure: If any critical check failed
    """
    all_passed = all(check.passed for check in checks)
    
    if not all_passed:
        failed = [check.name for check in checks if not check.passed]
        logger.error(f"QA Gate FAILED. Failed checks: {', '.join(failed)}")
        raise QAGateFailure(f"QA checks failed: {failed}")
    
    logger.info("✓ QA Gate PASSED")
    return True


class QAGateFailure(Exception):
    """Raised when QA gate fails."""
    pass
