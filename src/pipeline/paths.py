"""Centralized path management for the pipeline."""

from pathlib import Path
from typing import Optional


class PathBuilder:
    """Builds and manages all file paths used in the pipeline."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize path builder.
        
        Args:
            root_dir: Root directory of the project. Defaults to project root.
        """
        if root_dir is None:
            # Assume we're in src/pipeline, go up two levels
            root_dir = Path(__file__).parent.parent.parent
        
        self.root = Path(root_dir)
        self.config = self.root / "config"
        self.data = self.root / "data"
        self.outputs = self.root / "outputs"
        self.models = self.root / "models"
        self.reports = self.root / "reports"
        self.report = self.root / "report"
    
    # Data paths
    def raw_data(self, filename: str) -> Path:
        """Path to raw data file."""
        return self.data / "raw" / filename
    
    def clean_data(self, filename: str) -> Path:
        """Path to cleaned data file."""
        return self.data / "clean" / filename
    
    def feature_data(self, filename: str) -> Path:
        """Path to feature data file."""
        return self.data / "features" / filename
    
    # Output paths
    def baseline_pred(self, filename: str) -> Path:
        """Path to baseline prediction output."""
        return self.outputs / "preds_baseline" / filename
    
    def model_pred(self, filename: str) -> Path:
        """Path to model prediction output."""
        return self.outputs / "preds_model" / filename
    
    def signal(self, filename: str) -> Path:
        """Path to trading signal output."""
        return self.outputs / "signals" / filename
    
    # Model paths
    def trained_model(self, filename: str) -> Path:
        """Path to trained model artifact."""
        return self.models / "trained" / filename
    
    # Report paths
    def qa_report(self, filename: str) -> Path:
        """Path to QA report."""
        return self.reports / "qa" / filename
    
    def metrics_report(self, filename: str) -> Path:
        """Path to metrics report."""
        return self.reports / "metrics" / filename
    
    def validation_report(self, filename: str) -> Path:
        """Path to validation report."""
        return self.reports / "validation" / filename
    
    def trading_report(self, filename: str) -> Path:
        """Path to trading report."""
        return self.reports / "trading" / filename
    
    def figure(self, filename: str) -> Path:
        """Path to figure file."""
        return self.reports / "figures" / filename
    
    def llm_log(self, filename: str) -> Path:
        """Path to LLM interaction log."""
        return self.reports / "llm_logs" / filename
    
    # Config paths
    def config_file(self, filename: str) -> Path:
        """Path to config file."""
        return self.config / filename
    
    def ensure_dirs(self):
        """Create all necessary directories."""
        dirs = [
            self.data / "raw",
            self.data / "clean",
            self.data / "features",
            self.outputs / "preds_baseline",
            self.outputs / "preds_model",
            self.outputs / "signals",
            self.models / "trained",
            self.reports / "qa",
            self.reports / "metrics",
            self.reports / "validation",
            self.reports / "trading",
            self.reports / "commentary",
            self.reports / "llm_logs",
            self.reports / "figures",
            self.report / "figures",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
