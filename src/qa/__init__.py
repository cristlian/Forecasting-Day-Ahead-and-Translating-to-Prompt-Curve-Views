"""QA package: Data quality checks and gate enforcement."""

from .checks import (
    CheckResult,
    run_all_checks,
    CompletenessCheck,
    DuplicateCheck,
    RangeCheck,
    TemporalCheck,
    HourlyContinuityCheck,
    AlignmentCheck,
    OutlierCheck,
)
from .gate import (
    QAGateFailure,
    evaluate_qa_gate,
    run_qa_pipeline,
    clean_dataset,
    get_qa_summary,
)
from .report import (
    generate_qa_report,
    format_qa_summary_for_log,
)

__all__ = [
    "CheckResult",
    "run_all_checks",
    "QAGateFailure",
    "evaluate_qa_gate",
    "run_qa_pipeline",
    "clean_dataset",
    "get_qa_summary",
    "generate_qa_report",
    "format_qa_summary_for_log",
]
