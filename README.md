# Power Price Forecasting Pipeline

A production-ready pipeline for forecasting European day-ahead power prices and generating actionable trading signals.

## Features

- **Forecasting Models**: Baseline (naive seasonal) and improved (LightGBM) models with 44% MAE improvement
- **Trading Agent**: LLM-powered signal generation with Clean Spark Spread analysis
- **Data Pipeline**: Automated ingestion from Energy-Charts with QA validation
- **Stress Testing**: Volatility, missing data, and regime-change robustness tests

## Quick Start

```bash
# Install
git clone https://github.com/cristlian/Forecasting-Day-Ahead-and-Translating-to-Prompt-Curve-Views.git
cd Forecasting-Day-Ahead-and-Translating-to-Prompt-Curve-Views
pip install -r requirements.txt

# Train models (uses local data in data/raw/)
python -m pipeline train

# Generate trading signal
python -m pipeline agent
```

## Commands

| Command | Description |
|---------|-------------|
| `python -m pipeline train` | Train baseline and improved models |
| `python -m pipeline train --use-sample` | Train with synthetic data (no external data needed) |
| `python -m pipeline validate --date 2026-01-29` | Run validation and stress tests |
| `python -m pipeline agent` | Generate morning trading signal via LLM |
| `python -m pipeline agent --no-llm` | Generate signal with rule-based fallback |
| `python -m pipeline eval` | Run cross-validation evaluation |
| `pytest tests/` | Run test suite (62 tests) |

## Project Structure

```
├── src/
│   ├── pipeline/      # CLI and orchestration
│   ├── models/        # Baseline and LightGBM models
│   ├── features/      # Feature engineering
│   ├── trading/       # Signal generation and trading agent
│   ├── validation/    # Metrics and stress tests
│   └── qa/            # Data quality checks
├── config/            # YAML configuration files
├── data/raw/          # Input data (Energy-Charts CSVs)
├── models/trained/    # Saved model artifacts
├── outputs/           # Predictions
├── reports/           # Metrics, validation reports, trading signals
└── tests/             # Unit tests
```

## Configuration

All settings are in `config/`:

- `market.yaml` - Target market (DE-LU), data sources
- `model.yaml` - Model hyperparameters, cross-validation settings
- `features.yaml` - Feature definitions, lag windows
- `reporting.yaml` - LLM provider settings (Gemini/OpenAI/Anthropic)

### Environment Variables

```bash
export GEMINI_API_KEY="your-key"  # Required for LLM trading agent
```

## Model Performance

Tested on DE-LU market data (12,847 hourly samples):

| Model | MAE (€/MWh) | RMSE (€/MWh) |
|-------|-------------|--------------|
| Baseline (naive seasonal) | 32.90 | 46.64 |
| Improved (LightGBM) | 18.38 | 26.78 |

**Improvement: 44% MAE reduction**

## Trading Signal Output

The `agent` command generates a 3-bullet execution strategy:

```
1. POSITION: BUY/SELL [bucket] [size]
2. RATIONALE: Clean Spark Spread analysis and forecast interpretation
3. RISK: Key invalidation conditions and monitoring points
```

Output saved to `reports/trading/LATEST_MORNING_SIGNAL.md`

## Data Sources

- **Prices**: Energy-Charts API (DE-LU day-ahead auction)
- **Load**: Day-ahead load forecasts
- **Generation**: Wind and solar generation forecasts

Data files expected in `data/raw/`:
- `prices_DE_LU.csv`
- `load_DE_LU.csv`
- `gen_forecast_DE_LU.csv`

## Leakage Prevention

All features use information available before the D+1 auction (12:00 CET):
- Minimum 24h lag on price-based features
- Rolling windows use shift-then-roll pattern
- Only day-ahead forecasts, not actuals

## Requirements

- Python 3.10+
- Dependencies: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `google-genai`

Full list in `requirements.txt`.

## License

MIT
