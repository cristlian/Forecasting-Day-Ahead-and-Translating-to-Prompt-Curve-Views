"""Assemble final report from pipeline outputs."""

import logging
from pathlib import Path
from typing import Dict, List
import json

logger = logging.getLogger(__name__)


def assemble_report(
    sections: Dict[str, str],
    output_path: Path,
    figures: List[Path] = None
) -> None:
    """
    Assemble final report from sections.
    
    Args:
        sections: Dictionary of section_name -> content
        output_path: Path to save final report
        figures: List of figure paths to include
    """
    logger.info("Assembling final report")
    
    # Load template
    template_path = Path(__file__).parent / "templates" / "report_sections.md.tpl"
    
    if template_path.exists():
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Format with sections
        report = template.format(**sections)
    else:
        # Fallback: concatenate sections
        report = _build_report_from_sections(sections)
    
    # Add figures if provided
    if figures:
        report += "\n\n## Figures\n\n"
        for fig_path in figures:
            report += f"\n![{fig_path.stem}]({fig_path})\n"
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_path}")


def _build_report_from_sections(sections: Dict[str, str]) -> str:
    """Build report by concatenating sections."""
    report_parts = []
    
    section_order = [
        "executive_summary",
        "objective",
        "scope",
        "data_sources",
        "feature_engineering",
        "model_architecture",
        "validation_approach",
        "qa_summary",
        "overall_metrics",
        "baseline_comparison",
        "feature_importance",
        "bucket_performance",
        "stress_tests",
        "robustness_analysis",
        "trading_signals",
        "conclusions",
        "configuration",
    ]
    
    for section_key in section_order:
        if section_key in sections:
            # Format section name
            title = section_key.replace("_", " ").title()
            report_parts.append(f"## {title}\n\n{sections[section_key]}\n")
    
    return "\n".join(report_parts)


def load_section_from_file(filepath: Path) -> str:
    """Load a section from a markdown file."""
    if not filepath.exists():
        logger.warning(f"Section file not found: {filepath}")
        return f"Section not available: {filepath.name}"
    
    with open(filepath, 'r') as f:
        return f.read()


def load_json_as_table(filepath: Path) -> str:
    """Load JSON data and format as markdown table."""
    if not filepath.exists():
        return "Data not available"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Simple table formatting
    if isinstance(data, dict):
        lines = ["| Metric | Value |", "|--------|-------|"]
        for key, value in data.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)
    
    return str(data)
