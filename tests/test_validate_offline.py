"""End-to-end validation tests (offline)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set matplotlib backend to non-GUI before any imports that might use it
import matplotlib
matplotlib.use("Agg")


def test_validate_offline_runs():
    from pipeline.config import load_config
    from validation.runner import run_validation

    config = load_config(Path(__file__).parent.parent / "config")
    result = run_validation(
        config=config,
        validation_date="2026-01-29",
        use_sample=True,
        cache_only=False,
        llm_test=False,
    )

    assert result.report_path.exists()
    assert result.metrics_path.exists()
    assert result.run_id.startswith("20260129_")
