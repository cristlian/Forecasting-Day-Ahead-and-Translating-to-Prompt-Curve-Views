"""Generate figures for the submission document.

Creates:
1. Feature importance bar chart
2. Model comparison (baseline vs improved) timeseries
3. Error distribution comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
import json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_run_id():
    """Find the most recent run ID from metrics files."""
    model_files = list(METRICS_DIR.glob("model_*.json"))
    if not model_files:
        raise FileNotFoundError("No model metrics found")
    
    # Sort by filename (timestamp)
    latest = sorted(model_files, key=lambda x: x.stem)[-1]
    # Extract run_id: model_YYYYMMDD_HHMMSS_MARKET.json -> YYYYMMDD_HHMMSS_MARKET
    run_id = latest.stem.replace("model_", "")
    return run_id


def plot_feature_importance(run_id: str):
    """Create feature importance bar chart."""
    fi_path = METRICS_DIR / f"feature_importance_{run_id}.csv"
    if not fi_path.exists():
        print(f"Feature importance file not found: {fi_path}")
        return
    
    df = pd.read_csv(fi_path)
    
    # Take top 10 features
    df_top = df.nlargest(10, "importance").sort_values("importance", ascending=True)
    
    # Normalize to percentage
    total = df["importance"].sum()
    df_top["importance_pct"] = df_top["importance"] / total * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(df_top["feature"], df_top["importance_pct"], color="#2563eb")
    ax.set_xlabel("Relative Importance (%)", fontsize=12)
    ax.set_title("Top 10 Feature Importance (LightGBM)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars, df_top["importance_pct"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f"{val:.1f}%", va="center", fontsize=10)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_model_comparison(run_id: str):
    """Create baseline vs improved model comparison."""
    # Load predictions
    baseline_path = OUTPUTS_DIR / "preds_baseline" / f"{run_id}.csv"
    model_path = OUTPUTS_DIR / "preds_model" / f"{run_id}.csv"
    
    if not baseline_path.exists() or not model_path.exists():
        print(f"Prediction files not found for {run_id}")
        return
    
    df_baseline = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
    df_model = pd.read_csv(model_path, index_col=0, parse_dates=True)
    
    # Take last 168 hours (1 week) for visualization
    df_baseline = df_baseline.tail(168)
    df_model = df_model.tail(168)
    
    # Align indices
    common_idx = df_baseline.index.intersection(df_model.index)
    df_baseline = df_baseline.loc[common_idx]
    df_model = df_model.loc[common_idx]
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Time series
    ax1 = axes[0]
    ax1.plot(df_model.index, df_model["actual"], label="Actual", color="#1f2937", linewidth=2)
    ax1.plot(df_baseline.index, df_baseline["predicted"], label="Baseline", color="#dc2626", alpha=0.7, linestyle="--")
    ax1.plot(df_model.index, df_model["predicted"], label="Improved", color="#2563eb", alpha=0.8)
    ax1.set_ylabel("Price (€/MWh)", fontsize=12)
    ax1.set_title("Model Predictions: Last 7 Days", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Errors
    ax2 = axes[1]
    baseline_error = df_baseline["actual"] - df_baseline["predicted"]
    model_error = df_model["actual"] - df_model["predicted"]
    
    ax2.fill_between(df_baseline.index, baseline_error, alpha=0.5, color="#dc2626", label="Baseline Error")
    ax2.fill_between(df_model.index, model_error, alpha=0.5, color="#2563eb", label="Improved Error")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("Prediction Error (€/MWh)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_error_distribution(run_id: str):
    """Create error distribution histogram comparison."""
    baseline_path = OUTPUTS_DIR / "preds_baseline" / f"{run_id}.csv"
    model_path = OUTPUTS_DIR / "preds_model" / f"{run_id}.csv"
    
    if not baseline_path.exists() or not model_path.exists():
        print(f"Prediction files not found for {run_id}")
        return
    
    df_baseline = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
    df_model = pd.read_csv(model_path, index_col=0, parse_dates=True)
    
    baseline_error = df_baseline["actual"] - df_baseline["predicted"]
    model_error = df_model["actual"] - df_model["predicted"]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(-100, 100, 51)
    ax.hist(baseline_error, bins=bins, alpha=0.6, color="#dc2626", label=f"Baseline (MAE: {baseline_error.abs().mean():.1f})")
    ax.hist(model_error, bins=bins, alpha=0.6, color="#2563eb", label=f"Improved (MAE: {model_error.abs().mean():.1f})")
    
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (€/MWh)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Error Distribution: Baseline vs Improved Model", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "error_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_metrics_summary(run_id: str):
    """Create metrics comparison bar chart."""
    baseline_path = METRICS_DIR / f"baseline_{run_id}.json"
    model_path = METRICS_DIR / f"model_{run_id}.json"
    
    if not baseline_path.exists() or not model_path.exists():
        print(f"Metrics files not found for {run_id}")
        return
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(model_path) as f:
        model = json.load(f)
    
    metrics = ["mae", "rmse"]
    baseline_vals = [baseline["metrics"][m] for m in metrics]
    model_vals = [model["metrics"][m] for m in metrics]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="#dc2626")
    bars2 = ax.bar(x + width/2, model_vals, width, label="Improved", color="#2563eb")
    
    ax.set_ylabel("€/MWh", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE", "RMSE"], fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{bar.get_height():.1f}", ha="center", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{bar.get_height():.1f}", ha="center", fontsize=10)
    
    # Add improvement annotation
    mae_improvement = (baseline_vals[0] - model_vals[0]) / baseline_vals[0] * 100
    ax.annotate(f"{mae_improvement:.0f}% improvement", 
                xy=(0, model_vals[0]), xytext=(0.5, model_vals[0] + 5),
                fontsize=11, fontweight="bold", color="#059669",
                arrowprops=dict(arrowstyle="->", color="#059669"))
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def copy_validation_figures():
    """Copy validation figures to report directory."""
    import shutil
    
    src_dir = PROJECT_ROOT / "reports" / "figures"
    for fig in src_dir.glob("validation_*.png"):
        dst = FIGURES_DIR / fig.name
        shutil.copy(fig, dst)
        print(f"Copied: {dst}")


def main():
    print("Generating figures for submission document...\n")
    
    try:
        run_id = find_latest_run_id()
        print(f"Using run_id: {run_id}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    plot_feature_importance(run_id)
    plot_model_comparison(run_id)
    plot_error_distribution(run_id)
    plot_metrics_summary(run_id)
    copy_validation_figures()
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
