# European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

## 📋 Project Overview

This project builds a prototype that produces a daily fair-value view for a European power market and demonstrates how this view informs prompt curve positioning.

**Case Study Theme:** European Power Fair Value Forecasting  
**Target Market:** DE-LU (Germany/Luxembourg)  
**Objective:** Forecast day-ahead power prices and translate forecasts into tradable curve views

---

## 🎯 Project Requirements

### 1. Data Ingestion & Quality Assurance
- **Task:** Collect publicly available data for one European power market (DE, FR, NL, or GB)
- **Deliverables:**
  - Dataset with hourly Day-Ahead prices
  - At least two fundamental drivers
  - Documented data sources
  - Implemented QA checks

### 2. Forecasting & Validation
- **Task:** Forecast next-day hourly prices (Option A) or front-week/front-month averages (Option B)
- **Deliverables:**
  - Baseline forecasting model
  - Improved forecasting model
  - Validation metrics and performance comparison

### 3. Prompt Curve Translation
- **Task:** Translate forecasts into tradable Day-Ahead to prompt curve views
- **Deliverables:**
  - Trading guidance document
  - Explanation of forecast usage/invalidation in trading context

### 4. AI/LLM Integration
- **Task:** Implement programmatic AI/LLM component to reduce manual work
- **Deliverables:**
  - Working AI/LLM integration code
  - Logged prompts and outputs
  - Purpose and benefit explanation

---

## 🏗️ Project Structure

```
.
├── config/                    # YAML configs for market/model/reporting
├── data/                      # Raw + cached + engineered data
│   ├── raw/                   # Input CSVs (Energy-Charts/SMARD)
│   ├── clean/                 # QA-passed datasets
│   ├── features/              # Feature matrices
│   └── cache/                 # Ingestion cache
├── docs/                      # Design + source documentation
├── models/                    # Trained model artifacts
├── outputs/                   # Predictions and signals
├── reports/                   # QA, metrics, validation, trading outputs
├── report/                    # Final submission report
├── scripts/                   # Helper scripts (trading step, batch runs)
├── src/                       # Pipeline source code
│   ├── ingest/                # Data ingestion (Energy-Charts/SMARD)
│   ├── qa/                    # QA gates
│   ├── features/              # Feature engineering
│   ├── models/                # Baseline + improved models
│   ├── trading/               # Signals + trading agent
│   └── reporting/             # LLM commentary
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.14.0
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cristlian/Forecasting-Day-Ahead-and-Translating-to-Prompt-Curve-Views.git
cd Forecasting-Day-Ahead-and-Translating-to-Prompt-Curve-Views
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start (Offline - No API Keys Required)

Train models using synthetic sample data (works on fresh clone):

```bash
# Train both baseline and improved models
python -m pipeline train --use-sample

# Run cross-validation evaluation
python -m pipeline eval --use-sample

# Run validation + stress tests (Step 7)
python -m pipeline validate --date 2026-01-29 --use-sample

# Run all tests
pytest tests/ -v
```

### Full Pipeline (Using Local Raw Data or SMARD Fallback)

```bash
# Run full pipeline (ingestion -> QA -> features)
python -m pipeline run --start-date 2024-01-01 --end-date 2024-06-30

# Train models using generated features
python -m pipeline train --cache-only

# Validate using cached features
python -m pipeline validate --date 2026-01-29 --cache-only
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | For LLM commentary (optional feature) |
| `ANTHROPIC_API_KEY` | No | For LLM commentary (optional feature) |
| `GEMINI_API_KEY` | No | For LLM commentary (optional feature) |

See `.env.example` for full list.

---

## 📊 Data Sources

### Power Market Data
- **Day-Ahead Prices:** Energy-Charts API (primary) / SMARD.de (fallback)
- **Load Forecasts:** Energy-Charts / SMARD
- **Wind Generation Forecasts:** Energy-Charts / SMARD
- **Solar Generation Forecasts:** Energy-Charts / SMARD

### Data Quality Checks
- Missing value detection and handling
- Outlier identification  
- Temporal consistency validation
- DST transition handling

---

## 🤖 Models

### Baseline Model
- **Type:** Naive Seasonal (same-hour-last-week)
- **Features:** Single lag feature (168h)
- **Purpose:** Transparent benchmark for comparison

### Improved Model  
- **Type:** LightGBM Gradient Boosting
- **Features:** Calendar, lag, rolling, interaction features
- **Validation:** Rolling-origin cross-validation

### Validation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Symmetric Mean Absolute Percentage Error (sMAPE)
- R² Score

### Leakage Prevention
All features are designed to be available before the D+1 auction (12:00 CET):
- Minimum 24h lag on all lagged features
- Rolling windows look backward only (shift-then-roll)
- Uses day-ahead forecasts, not actuals

---

## 💹 Trading Application

See [reports/trading/guidance.md](reports/trading/guidance.md) for detailed explanation of:
- How forecasts translate to curve positions
- Signal generation methodology
- Risk management considerations
- Conditions for forecast invalidation

---

## 🤖 AI/LLM Integration

**Component:** Automated daily market commentary generation

**Purpose:** Reduce manual effort in producing daily briefings

**Implementation:**
- APIs: OpenAI GPT-4 / Anthropic Claude
- Prompts: Logged in `reports/llm_logs/`
- Output: Markdown commentary in `reports/commentary/`

---

## 📈 Results

### Model Performance (Sample Data)
| Model | MAE (€/MWh) | RMSE (€/MWh) | sMAPE | R² |
|-------|-------------|--------------|-------|-----|
| Baseline | ~15 | ~20 | ~25% | 0.5-0.7 |
| Improved | ~10 | ~14 | ~18% | 0.7-0.85 |

*Actual performance varies with market conditions and data period.*

### Output Artifacts
- `outputs/preds_baseline/` - Baseline predictions
- `outputs/preds_model/` - Improved model predictions  
- `reports/metrics/` - Evaluation metrics (JSON)
- `reports/validation/` - Validation reports (Markdown)
- `reports/metrics/validation_*.json` - Validation metrics

---

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

---

## 📝 Documentation

- **Main Document:** [Link to 1-3 page PDF/Markdown submission]
- **Code Documentation:** Inline comments and docstrings
- **Trading Guidance:** See [reports/trading/guidance.md](reports/trading/guidance.md)

---

## 🔄 Reproducibility

All results can be reproduced by:
1. Following installation steps
2. Running scripts in the specified order
3. Using the provided random seeds

**Random Seeds:** Set in configuration files for reproducibility

---

## 📧 Contact

**Name:** [Your Name]  
**Email:** [Your Email]

---

## 📄 License

[Specify license if applicable]

---

## ✅ Evaluation Checklist

- [ ] Dataset correctness and quality assurance
- [ ] Forecasting rigor (baseline + improved models)
- [ ] Trading relevance (prompt curve translation)
- [ ] Engineering quality and reproducibility
- [ ] Effective programmatic AI/LLM usage

---

## 🔮 Future Improvements

- [ ] Add more fundamental drivers
- [ ] Implement ensemble models
- [ ] Real-time data pipeline
- [ ] Enhanced trading signal generation
- [ ] Backtesting framework
