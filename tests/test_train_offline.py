"""Tests for offline model training (no API keys required)."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestOfflineTraining:
    """Tests that verify training works without API keys."""
    
    def test_sample_data_generation(self):
        """Test that sample data can be generated."""
        from data.sample import generate_sample_features
        
        df = generate_sample_features(n_days=30, seed=42)
        
        # Check basic properties
        assert len(df) > 0
        assert "day_ahead_price" in df.columns
        assert "forecast_load" in df.columns
        assert "lag_24h_day_ahead_price" in df.columns
        
        # Check for reasonable values
        assert df["day_ahead_price"].mean() > 0
        assert df["forecast_load"].mean() > 10000
    
    def test_sample_data_deterministic(self):
        """Test that sample data generation is deterministic with seed."""
        from data.sample import generate_sample_features
        
        df1 = generate_sample_features(n_days=10, seed=123)
        df2 = generate_sample_features(n_days=10, seed=123)
        
        assert df1["day_ahead_price"].iloc[0] == df2["day_ahead_price"].iloc[0]
    
    def test_baseline_training_with_sample_data(self):
        """Test baseline model training with sample data."""
        from data.sample import generate_sample_features
        from models.baseline import train_baseline, NaiveSeasonalModel
        from validation.metrics import calculate_metrics
        
        # Generate sample data
        df = generate_sample_features(n_days=30, seed=42)
        
        # Create minimal config
        config = {
            "model": {
                "baseline": {"lag_hours": 168}
            },
            "market": {
                "target": {"column": "day_ahead_price"}
            }
        }
        
        # Train baseline
        model, predictions_df = train_baseline(
            df_features=df,
            config=config,
            target_col="day_ahead_price",
        )
        
        # Verify predictions
        assert len(predictions_df) > 0
        assert "actual" in predictions_df.columns
        assert "predicted" in predictions_df.columns
        
        # Check metrics are reasonable
        metrics = calculate_metrics(
            predictions_df["actual"].values,
            predictions_df["predicted"].values,
        )
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE always
    
    def test_improved_model_training_with_sample_data(self):
        """Test improved model training with sample data."""
        from data.sample import generate_sample_features
        from models.model import train_improved_model, PowerPriceModel
        from validation.metrics import calculate_metrics
        
        # Generate sample data  
        df = generate_sample_features(n_days=30, seed=42)
        
        # Create minimal config
        config = {
            "model": {
                "improved_model": {
                    "type": "lightgbm",
                    "hyperparameters": {
                        "n_estimators": 50,  # Small for fast testing
                        "learning_rate": 0.1,
                        "max_depth": 4,
                        "random_state": 42,
                    }
                }
            },
            "market": {
                "target": {"column": "day_ahead_price"}
            }
        }
        
        # Train model
        model, train_preds, val_preds = train_improved_model(
            df_features=df,
            config=config,
            target_col="day_ahead_price",
        )
        
        # Verify predictions
        assert len(val_preds) > 0
        assert "actual" in val_preds.columns
        assert "predicted" in val_preds.columns
        
        # Check metrics
        metrics = calculate_metrics(
            val_preds["actual"].values,
            val_preds["predicted"].values,
        )
        assert metrics["mae"] >= 0
        
        # Verify feature importance exists
        assert model.feature_importance is not None
        assert len(model.feature_importance) > 0
    
    def test_cv_evaluation_with_sample_data(self):
        """Test cross-validation with sample data."""
        from data.sample import generate_sample_features
        from models.cv import RollingOriginCV
        from models.model import PowerPriceModel
        
        # Generate enough data for CV
        df = generate_sample_features(n_days=60, seed=42)
        
        config = {
            "model": {
                "improved_model": {
                    "hyperparameters": {
                        "n_estimators": 20,
                        "max_depth": 3,
                        "random_state": 42,
                    }
                }
            }
        }
        
        # Setup CV with small splits for testing
        cv = RollingOriginCV(
            train_size_days=20,
            test_size_days=5,
            n_splits=2,
        )
        
        # Run CV
        results = cv.evaluate(
            df=df,
            model_class=PowerPriceModel,
            config=config,
            target_col="day_ahead_price",
        )
        
        # Verify results
        assert "mean_metrics" in results
        assert "split_metrics" in results
        assert len(results["split_metrics"]) >= 1
        assert results["mean_metrics"]["mae"] >= 0


class TestTrainingPipeline:
    """Tests for the full training pipeline."""
    
    def test_load_features_with_sample(self):
        """Test loading features with --use-sample flag."""
        from pipeline.train import load_features
        from pipeline.paths import PathBuilder
        
        paths = PathBuilder()
        
        df, source = load_features(
            paths=paths,
            cache_only=False,
            use_sample=True,
        )
        
        assert len(df) > 0
        assert source == "sample"
        assert "day_ahead_price" in df.columns
    
    def test_run_training_with_sample(self):
        """Test full training run with sample data."""
        from pipeline.train import run_training
        from pipeline.config import load_config
        
        config = load_config("config")
        
        results = run_training(
            config=config,
            model_type="both",
            cache_only=False,
            use_sample=True,
        )
        
        # Verify both models trained
        assert "baseline" in results
        assert "improved" in results
        
        # Check baseline succeeded
        assert results["baseline"].success
        assert results["baseline"].metrics is not None
        assert results["baseline"].metrics.get("mae", -1) >= 0
        
        # Check improved model succeeded
        assert results["improved"].success
        assert results["improved"].metrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
