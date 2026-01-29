"""Validation runner for Step 7: metrics + stress tests + report."""

import logging
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.paths import PathBuilder
from pipeline.train import CacheMissingError, load_features
from models.baseline import NaiveSeasonalModel
from models.model import PowerPriceModel
from validation.metrics import (
    calculate_metrics,
    calculate_bucket_metrics,
    define_hour_buckets,
    calculate_baseline_improvement,
)
from validation.plots import plot_predictions_vs_actual, plot_scatter
from validation.stress_tests import (
    test_missing_driver_robustness,
    test_volatility_performance,
    test_weekday_weekend_performance,
)
from features.build import validate_no_leakage
from reporting.llm_commentary import generate_commentary

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    run_id: str
    report_path: Path
    metrics_path: Path
    figures: Dict[str, Path]
    metrics: Dict[str, Any]
    stress_tests: Dict[str, Any]
    leakage_checks: Dict[str, Any]
    llm_test: Dict[str, Any]


def _deterministic_run_id(market: str, validation_date: datetime) -> str:
    return f"{validation_date.strftime('%Y%m%d')}_{market}"


def _split_train_validation(
    df: pd.DataFrame,
    validation_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    last_ts = df.index.max()
    if pd.isna(last_ts):
        raise ValueError("Feature data has no timestamps")
    cutoff = last_ts - pd.Timedelta(days=validation_days)
    df_train = df.loc[df.index < cutoff]
    df_val = df.loc[df.index >= cutoff]
    return df_train, df_val


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "smape": float("nan"), "r2": float("nan")}
    return calculate_metrics(y_true, y_pred)


def _leakage_sanity_checks(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    issues = []

    # Check ordering
    if not df.index.is_monotonic_increasing:
        issues.append("Timestamps are not strictly increasing")

    # Check lag feature naming and minimum lag
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    bad_lags = []
    for col in lag_cols:
        try:
            lag_hours = int(col.split("lag_")[1].split("h_")[0])
            if lag_hours < 24:
                bad_lags.append(col)
        except (IndexError, ValueError):
            continue
    if bad_lags:
        issues.append(f"Lag features with <24h shift: {bad_lags}")

    # Check for suspicious future/lead columns
    suspicious = [c for c in df.columns if any(x in c.lower() for x in ["lead", "future", "t+", "shift_-", "ahead_"])]
    if suspicious:
        issues.append(f"Suspicious feature names: {suspicious}")

    leakage_ok = validate_no_leakage(df, config)

    return {
        "leakage_ok": leakage_ok and len(issues) == 0,
        "issues": issues,
    }


def _format_metrics_table(baseline: Dict[str, float], improved: Dict[str, float]) -> str:
    improvement = calculate_baseline_improvement(baseline, improved)
    rows = [
        "| Metric | Baseline | Improved | Improvement |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["mae", "rmse", "smape", "r2"]:
        base_val = baseline.get(metric, float("nan"))
        imp_val = improved.get(metric, float("nan"))
        if metric in improvement:
            delta = improvement.get(f"{metric}_improvement_pct", float("nan"))
            delta_str = f"{delta:.2f}%"
        else:
            delta_str = "n/a"
        rows.append(f"| {metric.upper()} | {base_val:.4f} | {imp_val:.4f} | {delta_str} |")
    return "\n".join(rows)


def _format_bucket_table(bucket_metrics: Dict[str, Dict[str, float]], metric: str = "mae") -> str:
    rows = [
        "| Bucket | Samples | " + metric.upper() + " |",
        "|---|---:|---:|",
    ]
    for bucket, metrics in bucket_metrics.items():
        rows.append(f"| {bucket} | {metrics.get('n_samples', 0)} | {metrics.get(metric, float('nan')):.4f} |")
    return "\n".join(rows)


def run_validation(
    config: Dict[str, Any],
    validation_date: Optional[str] = None,
    cache_only: bool = False,
    use_sample: bool = False,
    root_dir: Optional[Path] = None,
    llm_test: bool = False,
) -> ValidationResult:
    """Run validation and stress tests without ingestion."""
    paths = PathBuilder(root_dir)
    paths.ensure_dirs()

    market = config.get("market", {}).get("market", {}).get("code", "DE_LU")
    target_col = config.get("market", {}).get("target", {}).get("column", "day_ahead_price")

    if validation_date is None:
        validation_dt = datetime.utcnow()
    else:
        validation_dt = datetime.strptime(validation_date, "%Y-%m-%d")

    run_id = _deterministic_run_id(market, validation_dt)

    logger.info(f"Starting validation run: {run_id}")
    logger.info(f"Cache only: {cache_only}")
    logger.info(f"Use sample: {use_sample}")

    # Load features
    df_features, source = load_features(
        paths=paths,
        cache_only=cache_only,
        use_sample=use_sample,
    )
    logger.info(f"Loaded {len(df_features)} samples from {source}")

    if target_col not in df_features.columns:
        raise ValueError(f"Target column '{target_col}' not found in features")

    # Ensure baseline lag feature exists
    baseline_config = config.get("model", {}).get("baseline", {})
    lag_hours = baseline_config.get("lag_hours", 168)
    lag_col = f"lag_{lag_hours}h_{target_col}"
    if lag_col not in df_features.columns:
        df_features = df_features.copy()
        df_features[lag_col] = df_features[target_col].shift(lag_hours)

    # Filter rows with target
    df_features = df_features[df_features[target_col].notna()].sort_index()

    # Train/validation split
    validation_days = config.get("market", {}).get("split", {}).get("validation_days", 30)
    df_train, df_val = _split_train_validation(df_features, validation_days)

    if df_train.empty or df_val.empty:
        raise ValueError(
            "Insufficient data for validation split. "
            "Try reducing validation_days or use sample data."
        )

    # Baseline predictions
    df_val_baseline = df_val[df_val[lag_col].notna()].copy()
    if df_val_baseline.empty:
        raise ValueError("No valid validation samples for baseline (missing lagged values)")

    baseline_model = NaiveSeasonalModel(lag_hours=lag_hours, config=config)
    X_val_baseline = df_val_baseline.drop(columns=[target_col])
    y_val_baseline = df_val_baseline[target_col].values
    baseline_pred = baseline_model.predict(X_val_baseline)
    baseline_metrics = _safe_metrics(y_val_baseline, baseline_pred)

    # Improved model
    df_train_model = df_train.dropna()
    df_val_model = df_val.dropna()

    if df_train_model.empty or df_val_model.empty:
        raise ValueError("Not enough non-null samples for improved model validation")

    X_train = df_train_model.drop(columns=[target_col])
    y_train = df_train_model[target_col]
    X_val = df_val_model.drop(columns=[target_col])
    y_val = df_val_model[target_col]

    improved_model = PowerPriceModel(config)
    improved_model.fit(X_train, y_train)
    improved_pred = improved_model.predict(X_val)
    improved_metrics = _safe_metrics(y_val.values, improved_pred)

    # Bucketed metrics by hour-of-day
    hour_buckets = define_hour_buckets()
    baseline_bucketed = calculate_bucket_metrics(
        y_true=y_val_baseline,
        y_pred=baseline_pred,
        timestamps=df_val_baseline.index,
        buckets=hour_buckets,
    )
    improved_bucketed = calculate_bucket_metrics(
        y_true=y_val.values,
        y_pred=improved_pred,
        timestamps=df_val_model.index,
        buckets=hour_buckets,
    )

    # Stress tests
    driver_columns = [d.get("name") for d in config.get("market", {}).get("drivers", []) if d.get("name")]
    volatility_threshold = float(
        df_val_model[target_col].groupby(df_val_model.index.date).std().quantile(0.75)
    )

    stress_tests = {
        "weekday_weekend": test_weekday_weekend_performance(
            y_true=y_val.values,
            y_pred=improved_pred,
            timestamps=df_val_model.index,
        ),
        "volatility": test_volatility_performance(
            y_true=y_val.values,
            y_pred=improved_pred,
            timestamps=df_val_model.index,
            volatility_threshold=volatility_threshold,
        ),
        "missing_driver": test_missing_driver_robustness(
            model=improved_model,
            X_test=X_val,
            y_test=y_val,
            driver_columns=driver_columns,
        ),
    }

    leakage_checks = _leakage_sanity_checks(df_features, config)

    # Plots
    figures = {}
    fig_ts = paths.figure(f"validation_timeseries_{run_id}.png")
    plot_predictions_vs_actual(
        y_true=y_val.values,
        y_pred=improved_pred,
        timestamps=df_val_model.index,
        output_path=fig_ts,
        title=f"Validation Predictions vs Actual ({run_id})",
    )
    figures["timeseries"] = fig_ts

    fig_scatter = paths.figure(f"validation_scatter_{run_id}.png")
    plot_scatter(
        y_true=y_val.values,
        y_pred=improved_pred,
        output_path=fig_scatter,
        title=f"Validation Scatter ({run_id})",
    )
    figures["scatter"] = fig_scatter

    # Optional LLM test
    llm_test_result = {"ran": False, "status": "skipped"}
    if llm_test:
        llm_test_result["ran"] = True
        try:
            commentary = generate_commentary(
                context={
                    "forecast_summary": "Validation LLM test run",
                    "model_performance": f"Improved MAE: {improved_metrics.get('mae', float('nan')):.2f}",
                    "market_conditions": "Offline validation",
                    "key_drivers": ", ".join(driver_columns) or "N/A",
                    "stress_test_results": "See validation report",
                },
                config=config,
                output_path=None,
            )
            llm_test_result["status"] = commentary
        except Exception as exc:
            llm_test_result["status"] = f"LLM test failed: {exc}"

    # Save metrics
    metrics_path = paths.metrics_report(f"validation_{run_id}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "run_id": run_id,
        "validation_date": validation_dt.strftime("%Y-%m-%d"),
        "data_source": source,
        "train_samples": len(df_train_model),
        "validation_samples": len(df_val_model),
        "baseline": {
            "metrics": baseline_metrics,
            "bucketed": baseline_bucketed,
        },
        "improved": {
            "metrics": improved_metrics,
            "bucketed": improved_bucketed,
        },
        "stress_tests": stress_tests,
        "leakage_checks": leakage_checks,
        "llm_test": llm_test_result,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2, default=str)

    # Write report
    report_path = paths.validation_report(f"{run_id}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        f"# Validation Report - {run_id}",
        "",
        f"**Validation date:** {validation_dt.strftime('%Y-%m-%d')}",
        f"**Data source:** {source}",
        f"**Train samples:** {len(df_train_model)}",
        f"**Validation samples:** {len(df_val_model)}",
        "",
        "## Overall Metrics",
        _format_metrics_table(baseline_metrics, improved_metrics),
        "",
        "## Hour-of-Day Bucketed MAE (Improved)",
        _format_bucket_table(improved_bucketed, metric="mae"),
        "",
        "## Stress Tests",
        "### Weekday vs Weekend",
        f"Weekday MAE: {stress_tests['weekday_weekend']['weekday']['mae']:.4f}",
        f"Weekend MAE: {stress_tests['weekday_weekend']['weekend']['mae']:.4f}",
        "",
        "### High vs Low Volatility Days",
        f"High-volatility MAE: {stress_tests['volatility']['high_volatility']['mae']:.4f}",
        f"Low-volatility MAE: {stress_tests['volatility']['low_volatility']['mae']:.4f}",
        "",
        "### Missing Driver Robustness",
    ]

    for key, result in stress_tests["missing_driver"].items():
        if key == "baseline":
            continue
        if "mae_degradation_pct" in result:
            report_lines.append(f"- {key}: MAE degradation {result['mae_degradation_pct']:.2f}%")
        else:
            report_lines.append(f"- {key}: {result.get('error', 'n/a')}")

    report_lines.extend(
        [
            "",
            "## Leakage Sanity Checks",
            f"Leakage OK: {leakage_checks['leakage_ok']}",
        ]
    )

    if leakage_checks.get("issues"):
        report_lines.append("Issues:")
        report_lines.extend([f"- {issue}" for issue in leakage_checks["issues"]])

    if llm_test:
        report_lines.extend(
            [
                "",
                "## LLM Test",
                f"Status: {llm_test_result['status']}",
            ]
        )

    report_path.write_text("\n".join(report_lines))

    logger.info(f"Validation report saved to {report_path}")
    logger.info(f"Metrics saved to {metrics_path}")

    return ValidationResult(
        run_id=run_id,
        report_path=report_path,
        metrics_path=metrics_path,
        figures=figures,
        metrics={"baseline": baseline_metrics, "improved": improved_metrics},
        stress_tests=stress_tests,
        leakage_checks=leakage_checks,
        llm_test=llm_test_result,
    )
