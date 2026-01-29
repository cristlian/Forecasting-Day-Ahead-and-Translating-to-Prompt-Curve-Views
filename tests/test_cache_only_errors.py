"""Tests for cache-only error handling."""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCacheOnlyErrors:
    """Tests that verify helpful error messages when cache is missing."""
    
    def test_cache_missing_raises_helpful_error(self):
        """Test that cache-only mode produces helpful error when cache is missing."""
        from pipeline.train import load_features, CacheMissingError
        from pipeline.paths import PathBuilder
        
        # Create a temporary directory with no cache
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a PathBuilder pointing to empty directory
            paths = PathBuilder(root_dir=Path(tmpdir))
            paths.ensure_dirs()
            
            # Should raise CacheMissingError with helpful message
            with pytest.raises(CacheMissingError) as exc_info:
                load_features(
                    paths=paths,
                    cache_only=True,
                    use_sample=False,
                )
            
            # Check error message is helpful
            error_msg = str(exc_info.value)
            assert "No cached features found" in error_msg or "No features available" in error_msg
            assert "--use-sample" in error_msg  # Should suggest alternative
    
    def test_cache_only_with_existing_cache_works(self):
        """Test that cache-only works when cache exists."""
        from pipeline.train import load_features
        from pipeline.paths import PathBuilder
        from data.sample import generate_sample_features
        
        # Create a temporary directory with cache
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = PathBuilder(root_dir=Path(tmpdir))
            paths.ensure_dirs()
            
            # Create a fake cached features file
            run_id = "20260101_120000_DE_LU"
            features_dir = paths.data / "features" / run_id
            features_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save sample features to cache location
            df = generate_sample_features(n_days=10)
            df.to_parquet(features_dir / "features.parquet")
            
            # Should work with cache-only
            df_loaded, source = load_features(
                paths=paths,
                cache_only=True,
                use_sample=False,
            )
            
            assert len(df_loaded) > 0
            assert "cache:" in source
    
    def test_use_sample_always_works(self):
        """Test that --use-sample always works regardless of cache state."""
        from pipeline.train import load_features
        from pipeline.paths import PathBuilder
        
        # Even with empty temp directory, use_sample should work
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = PathBuilder(root_dir=Path(tmpdir))
            paths.ensure_dirs()
            
            # Create sample data directory
            sample_dir = paths.data / "sample"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Should work because it generates sample data
            df, source = load_features(
                paths=paths,
                cache_only=False,
                use_sample=True,
            )
            
            assert len(df) > 0
            assert source == "sample"


class TestErrorMessages:
    """Tests for clear error messages."""
    
    def test_missing_lag_column_error(self):
        """Test helpful error when lag column is missing."""
        import pandas as pd
        import numpy as np
        from models.baseline import NaiveSeasonalModel
        
        # Create data WITHOUT required lag column
        df = pd.DataFrame({
            "other_column": np.random.rand(100),
        })
        
        model = NaiveSeasonalModel(lag_hours=168)
        
        with pytest.raises(ValueError) as exc_info:
            model.predict(df)
        
        error_msg = str(exc_info.value)
        assert "lag_168h_day_ahead_price" in error_msg
        assert "not found" in error_msg
    
    def test_model_not_fitted_error(self):
        """Test helpful error when model is used before fitting."""
        import pandas as pd
        import numpy as np
        from models.model import PowerPriceModel
        
        config = {"model": {"improved_model": {"hyperparameters": {}}}}
        model = PowerPriceModel(config)
        
        df = pd.DataFrame({
            "feature1": np.random.rand(10),
        })
        
        with pytest.raises(ValueError) as exc_info:
            model.predict(df)
        
        assert "not trained" in str(exc_info.value).lower()


class TestAPIKeyIndependence:
    """Tests that verify no API keys are required for core functionality."""
    
    def test_no_env_vars_needed_for_sample_training(self, monkeypatch):
        """Test that training with sample data doesn't require any env vars."""
        # Clear all potentially relevant environment variables
        for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            monkeypatch.delenv(var, raising=False)
        
        from data.sample import generate_sample_features
        from models.baseline import train_baseline
        
        df = generate_sample_features(n_days=20, seed=42)
        
        config = {
            "model": {"baseline": {"lag_hours": 168}},
            "market": {"target": {"column": "day_ahead_price"}}
        }
        
        # Should work without any API keys
        model, predictions = train_baseline(df, config)
        
        assert len(predictions) > 0
    
    def test_config_loading_no_api_required(self, monkeypatch):
        """Test that config loading doesn't require API keys."""
        # Clear environment variables
        for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            monkeypatch.delenv(var, raising=False)
        
        from pipeline.config import load_config
        
        # Should work without API keys
        config = load_config("config")
        
        assert "market" in config
        assert "model" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
