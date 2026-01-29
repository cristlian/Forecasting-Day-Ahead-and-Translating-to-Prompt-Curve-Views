# Pipeline Progress Tracker

## Current Status: Steps 1-9 Complete ✅

### Completed
- [x] Scope freeze normalized into configs (market.yaml, schema.yaml, features.yaml)
- [x] docs/00_scope.md updated with DE-LU market details
- [x] **Data Source**: Energy-Charts API (Fraunhofer ISE) - No API key required!
- [x] Ingestion module (src/ingest/) with Energy-Charts + ENTSO-E fallback
- [x] QA gate (src/qa/) with checks and report generation
- [x] Feature pipeline (src/features/) with leakage-safe design
- [x] Pipeline wiring (src/pipeline/) with CLI
- [x] **Baseline model** (src/models/baseline.py) - Naive seasonal (lag=168h)
- [x] **Improved model** (src/models/model.py) - LightGBM gradient boosting
- [x] **Rolling-origin CV** (src/models/cv.py) - Time-series cross-validation
- [x] **Offline mode** - Works without API keys using sample data
- [x] **Step 7: Validation + Stress Tests** (src/validation/runner.py, tests/)
- [x] **Step 8: Trading Signals with Clean Spark Spread** (src/trading/signals.py)
- [x] **Step 9: Trading Agent** (src/trading/agent.py) - Senior Trader LLM persona

### All Tests Passing: 62 tests ✅

---

## How to Run

### Step 1: Download Data (Energy-Charts - No API Key!)

```bash
# Download 2 years of data automatically
python -c "from test_energy_charts import fetch_full_dataset; fetch_full_dataset('2023-01-01', '2024-12-31', 'data/raw')"
```

### Step 2: Train Models

```bash
# Train both models using synthetic sample data
python -m pipeline train --use-sample

# Train only baseline model
python -m pipeline train --model baseline --use-sample

# Train only improved model  
python -m pipeline train --model improved --use-sample

# Run cross-validation evaluation
python -m pipeline eval --use-sample
```

### Step 3: Run Validation + Stress Tests

```bash
# Run validation with date-specific deterministic run_id
python -m pipeline validate --use-sample --date 2026-01-29
```

### Step 4: Generate Trading Signals (Step 8)

```bash
# Generate trading signals with Clean Spark Spread
python scripts/run_trading_step.py

# Outputs:
# - reports/trading/hourly_signals.csv
# - reports/trading/signal_report.md
# - reports/trading/prompt_view.csv
```

### Step 5: Morning Trading Signal (Step 9 - Agent)

```bash
# Generate morning trading signal with Senior Trader LLM persona
python -m pipeline agent --use-sample

# Without LLM (rule-based fallback)
python -m pipeline agent --use-sample --no-llm

# Outputs:
# - reports/trading/LATEST_MORNING_SIGNAL.md
# - reports/trading/morning_signal_YYYYMMDD_HHMMSS.json
```

### Cache Mode (Requires Previous Pipeline Run)

```bash
# Train using cached features (if you've run the full pipeline before)
python -m pipeline train --cache-only

# This will fail with helpful message if no cache exists
```

### Full Pipeline (Requires ENTSOE_API_KEY or Uses SMARD Fallback)

```bash
# Set API key (optional - will use SMARD fallback if not set)
export ENTSOE_API_KEY='your-key-here'  # Linux/Mac
$env:ENTSOE_API_KEY='your-key-here'    # PowerShell

# Run full pipeline (ingestion -> QA -> features)
python -m pipeline run --start-date 2024-01-01 --end-date 2024-12-31

# Then train models on the generated features
python -m pipeline train --cache-only
```

### Running Tests

```bash
# Run all tests (no API keys needed) - 62 tests
pytest tests/ -v

# Run only offline training tests
pytest tests/test_train_offline.py -v

# Run trading agent tests
pytest tests/test_trading_agent.py -v
```

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ENTSOE_API_KEY` | No* | ENTSO-E Transparency Platform API for fresh data |
| `OPENAI_API_KEY` | No | LLM commentary generation (optional) |
| `ANTHROPIC_API_KEY` | No | LLM commentary generation (optional) |

\* Not required when using `--use-sample` or `--cache-only` modes.

See `.env.example` for details.

---

## Output Artifacts

### Data (gitignored)
- `data/raw/{run_id}/` - Raw ingested data (prices, fundamentals)
- `data/clean/{run_id}/` - QA-cleaned aligned dataset
- `data/features/{run_id}/` - Feature matrix for modeling
- `data/cache/` - API response cache
- `data/sample/` - Generated sample data for offline testing

### Model Outputs (gitignored)
- `outputs/preds_baseline/{run_id}.csv` - Baseline model predictions
- `outputs/preds_model/{run_id}.csv` - Improved model predictions
- `models/trained/model_{run_id}.bin` - Trained model artifact

### Reports
- `reports/qa/{run_id}_qa.md` - Human-readable QA report
- `reports/metrics/baseline_{run_id}.json` - Baseline model metrics
- `reports/metrics/model_{run_id}.json` - Improved model metrics
- `reports/metrics/feature_importance_{run_id}.csv` - Feature importance
- `reports/metrics/cv_results_{run_id}.json` - Cross-validation results
- `reports/validation/{run_id}.md` - Validation comparison report
- `reports/trading/hourly_signals.csv` - Hourly trading signals with CSS
- `reports/trading/signal_report.md` - Signal summary report
- `reports/trading/LATEST_MORNING_SIGNAL.md` - AI-generated morning signal

---

## Model Performance

### Baseline Model (Naive Seasonal)
- Uses same-hour-last-week (168h lag) as prediction
- Transparent, interpretable benchmark
- No training required

### Improved Model (LightGBM)
- Gradient boosting with feature engineering
- Hyperparameters tuned for power price forecasting
- Rolling-origin cross-validation for robust evaluation

### Leakage Prevention
1. Time-based train/test split (no shuffling)
2. Minimum 24h lag on all features (D+1 auction constraint)
3. Rolling windows look backward only
4. Uses forecasts, not actuals, for fundamentals
