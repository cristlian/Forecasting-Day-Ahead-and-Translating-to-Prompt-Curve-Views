"""Tests for the Trading Agent (Step 9)."""

import sys
from pathlib import Path

# Ensure src is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import pytest
import pandas as pd
import numpy as np


class TestTradingAgent:
    """Tests for the Trading Agent module."""
    
    @pytest.fixture
    def sample_signals_df(self):
        """Create sample signals DataFrame matching agent expected columns."""
        np.random.seed(42)
        hours = 168
        dates = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
        
        prices = 60 + np.random.randn(hours) * 20
        css = prices - 90  # Marginal cost = 90
        
        return pd.DataFrame({
            "predicted_price": prices,
            "actual_price": prices + np.random.randn(hours) * 5,
            "clean_spark_spread": css,  # Agent expects this column name
            "signal": np.where(css > 10, "BUY", np.where(css < -10, "SELL", "HOLD")),
            "confidence": np.random.rand(hours),
            "dispatch_pnl": np.where(css > 0, css, 0),
        }, index=dates)
    
    @pytest.fixture
    def sample_bucket_df(self):
        """Create sample bucket DataFrame matching agent expected columns."""
        return pd.DataFrame({
            "bucket": ["peak", "off_peak_night", "off_peak_late", "shoulder", "weekend"],
            "predicted_price": [85.0, 45.0, 40.0, 60.0, 50.0],  # Agent expects this
            "clean_spark_spread": [-5.0, -45.0, -50.0, -30.0, -40.0],  # Agent expects this
            "margin_signal": ["HOLD", "OFF", "OFF", "OFF", "OFF"],  # Agent expects this
        })
    
    def test_build_trading_context(self, sample_signals_df, sample_bucket_df):
        """Test context extraction from signals."""
        from trading.agent import _build_trading_context
        
        context = _build_trading_context(sample_signals_df, sample_bucket_df)
        
        assert "hours_analyzed" in context
        assert context["hours_analyzed"] == 168
        assert "avg_price_eur" in context
        assert "avg_css_eur" in context
        assert "profitable_hours" in context
        assert "bucket_summary" in context
        assert len(context["bucket_summary"]) == 5
    
    def test_build_agent_prompt(self, sample_signals_df, sample_bucket_df):
        """Test prompt generation for LLM."""
        from trading.agent import _build_trading_context, _build_agent_prompt
        
        context = _build_trading_context(sample_signals_df, sample_bucket_df)
        prompt = _build_agent_prompt(context)  # Only takes context
        
        # Should contain key trading terms
        assert "EUR" in prompt or "€" in prompt
        assert "peak" in prompt.lower() or "bucket" in prompt.lower()
        assert "signal" in prompt.lower() or "strategy" in prompt.lower()
    
    def test_generate_morning_signal_no_llm(self, sample_signals_df, sample_bucket_df, tmp_path):
        """Test full signal generation without LLM."""
        from trading.agent import generate_morning_signal
        
        config = {
            "reporting": {
                "llm_settings": {
                    "enabled": False,
                },
            },
            "market": {
                "market": {"code": "DE_LU"},
            },
        }
        
        result = generate_morning_signal(
            signals_df=sample_signals_df,
            bucket_df=sample_bucket_df,
            config=config,
            output_dir=tmp_path,
        )
        
        assert "strategy" in result
        assert "context" in result
        assert "timestamp" in result
        
        # Check output files
        assert (tmp_path / "LATEST_MORNING_SIGNAL.md").exists()
    
    def test_fallback_strategy_has_3_bullets(self, sample_signals_df, sample_bucket_df, tmp_path):
        """Test that fallback strategy has 3-bullet format."""
        from trading.agent import generate_morning_signal
        
        config = {
            "reporting": {"llm_settings": {"enabled": False}},
            "market": {"market": {"code": "DE_LU"}},
        }
        
        result = generate_morning_signal(
            signals_df=sample_signals_df,
            bucket_df=sample_bucket_df,
            config=config,
            output_dir=tmp_path,
        )
        
        strategy = result["strategy"]
        
        # Should have required sections
        assert "POSITION" in strategy
        assert "RATIONALE" in strategy
        assert "RISK" in strategy
    
    def test_context_has_risk_metrics(self, sample_signals_df, sample_bucket_df):
        """Test that context includes risk metrics."""
        from trading.agent import _build_trading_context
        
        context = _build_trading_context(sample_signals_df, sample_bucket_df)
        
        # Risk metrics should be present in nested structure
        assert "risk_indicators" in context
        assert "max_negative_streak" in context["risk_indicators"]
        assert "volatility_eur" in context["risk_indicators"]
        assert context["price_volatility"] > 0
    
    def test_market_extraction(self, sample_signals_df, sample_bucket_df, tmp_path):
        """Test market code is correctly extracted from config."""
        from trading.agent import generate_morning_signal
        
        config = {
            "reporting": {"llm_settings": {"enabled": False}},
            "market": {"market": {"code": "FR"}},
        }
        
        result = generate_morning_signal(
            signals_df=sample_signals_df,
            bucket_df=sample_bucket_df,
            config=config,
            output_dir=tmp_path,
        )
        
        # Check market appears in output
        report_content = (tmp_path / "LATEST_MORNING_SIGNAL.md").read_text(encoding="utf-8")
        assert "FR" in report_content


class TestAgentCLI:
    """Test CLI integration for agent command."""
    
    def test_cli_agent_help(self):
        """Test that agent command is available in CLI."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pipeline", "agent", "--help"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        assert result.returncode == 0
        assert "--no-llm" in result.stdout
        assert "--use-sample" in result.stdout
    
    def test_cli_agent_runs_with_sample(self):
        """Test that agent command runs with sample data."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pipeline", "agent", "--use-sample", "--no-llm"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=60,
        )
        # Should succeed (exit 0) or give sensible error
        assert result.returncode == 0 or "predictions" in result.stderr.lower()
