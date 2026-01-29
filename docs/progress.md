# Pipeline Progress Tracker

## Current Status: Steps 2-6 Implementation Complete

### Completed
- [x] Scope freeze normalized into configs (market.yaml, schema.yaml, features.yaml)
- [x] docs/00_scope.md updated with DE-LU market details
- [x] Ingestion module (src/ingest/) with ENTSO-E + SMARD fallback
- [x] QA gate (src/qa/) with checks and report generation
- [x] Feature pipeline (src/features/) with leakage-safe design
- [x] Pipeline wiring (src/pipeline/) with CLI
- [x] **Baseline model** (src/models/baseline.py) - Naive seasonal (lag=168h)
- [x] **Improved model** (src/models/model.py) - LightGBM gradient boosting
- [x] **Rolling-origin CV** (src/models/cv.py) - Time-series cross-validation
- [x] **Offline mode** - Works without API keys using sample data

### Next Steps
1. Generate trading signals from predictions
2. LLM commentary integration
3. Final report assembly

---

## How to Run

### Offline Mode (No API Keys Required)

```bash
# Train both models using synthetic sample data
python -m src.pipeline.cli train --use-sample

# Train only baseline model
python -m src.pipeline.cli train --model baseline --use-sample

# Train only improved model  
python -m src.pipeline.cli train --model improved --use-sample

# Run cross-validation evaluation
python -m src.pipeline.cli eval --use-sample
```

### Cache Mode (Requires Previous Pipeline Run)

```bash
# Train using cached features (if you've run the full pipeline before)
python -m src.pipeline.cli train --cache-only

# This will fail with helpful message if no cache exists
```

### Full Pipeline (Requires ENTSOE_API_KEY or Uses SMARD Fallback)

```bash
# Set API key (optional - will use SMARD fallback if not set)
export ENTSOE_API_KEY='your-key-here'  # Linux/Mac
$env:ENTSOE_API_KEY='your-key-here'    # PowerShell

# Run full pipeline (ingestion -> QA -> features)
python -m src.pipeline.cli run --start-date 2024-01-01 --end-date 2024-12-31

# Then train models on the generated features
python -m src.pipeline.cli train --cache-only
```

### Running Tests

```bash
# Run all tests (no API keys needed)
pytest tests/ -v

# Run only offline training tests
pytest tests/test_train_offline.py -v

# Run cache error handling tests
pytest tests/test_cache_only_errors.py -v
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
