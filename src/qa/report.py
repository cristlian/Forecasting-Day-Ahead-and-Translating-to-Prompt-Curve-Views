"""QA report generation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def generate_qa_report(
    checks: List,
    output_dir: Path,
    timestamp: datetime = None
) -> None:
    """
    Generate QA report in JSON and Markdown formats.
    
    Args:
        checks: List of QACheck results
        output_dir: Directory to save reports
        timestamp: Timestamp for report filename
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate JSON report
    json_path = output_dir / f"qa_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    _write_json_report(checks, json_path)
    
    # Generate Markdown report
    md_path = output_dir / f"qa_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    _write_md_report(checks, md_path, timestamp)
    
    logger.info(f"QA reports saved to {output_dir}")


def _write_json_report(checks: List, filepath: Path) -> None:
    """Write QA report as JSON."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "errors": check.errors,
                "warnings": check.warnings,
            }
            for check in checks
        ],
        "overall_pass": all(check.passed for check in checks),
    }
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)


def _write_md_report(checks: List, filepath: Path, timestamp: datetime) -> None:
    """Write QA report as Markdown."""
    lines = [
        f"# QA Report",
        f"",
        f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Summary",
        f"",
    ]
    
    # Summary table
    lines.append("| Check | Status | Errors | Warnings |")
    lines.append("|-------|--------|--------|----------|")
    
    for check in checks:
        status = "✓ PASS" if check.passed else "✗ FAIL"
        lines.append(f"| {check.name} | {status} | {len(check.errors)} | {len(check.warnings)} |")
    
    lines.append("")
    
    # Overall result
    overall = all(check.passed for check in checks)
    if overall:
        lines.append("**Overall Result:** ✓ PASSED")
    else:
        lines.append("**Overall Result:** ✗ FAILED")
    
    lines.append("")
    lines.append("## Details")
    lines.append("")
    
    # Details for each check
    for check in checks:
        lines.append(f"### {check.name}")
        lines.append("")
        
        if check.errors:
            lines.append("**Errors:**")
            for error in check.errors:
                lines.append(f"- {error}")
            lines.append("")
        
        if check.warnings:
            lines.append("**Warnings:**")
            for warning in check.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        if not check.errors and not check.warnings:
            lines.append("No issues found.")
            lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
