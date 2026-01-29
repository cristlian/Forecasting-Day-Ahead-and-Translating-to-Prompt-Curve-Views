# European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

## 📋 Project Overview

This project builds a prototype that produces a daily fair-value view for a European power market and demonstrates how this view informs prompt curve positioning.

**Case Study Theme:** European Power Fair Value Forecasting  
**Target Market:** [DE/FR/NL/GB - To be specified]  
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
├── data/                      # Raw and processed data
│   ├── raw/                   # Original datasets
│   └── processed/             # Cleaned and transformed data
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   ├── models/                # Model implementations
│   ├── utils/                 # Utility functions
│   └── ai_integration/        # AI/LLM components
├── outputs/                   # Generated outputs
│   ├── figures/               # Visualizations
│   ├── tables/                # Results tables
│   └── qa_reports/            # Quality assurance reports
├── docs/                      # Documentation
│   └── trading_guidance.md    # Prompt curve translation guide
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── submission.csv             # Out-of-sample predictions (optional)
└── README.md                  # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
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

### Usage

**Step 1: Data Collection**
```bash
python src/data/collect_data.py --market [DE/FR/NL/GB]
```

**Step 2: Run Quality Assurance**
```bash
python src/data/quality_assurance.py
```

**Step 3: Train Models**
```bash
python src/models/train.py --model baseline
python src/models/train.py --model improved
```

**Step 4: Generate Predictions**
```bash
python src/models/predict.py --output submission.csv
```

**Step 5: Run AI/LLM Component**
```bash
python src/ai_integration/run_ai_pipeline.py
```

---

## 📊 Data Sources

### Power Market Data
- **Day-Ahead Prices:** [Source TBD]
- **Fundamental Driver 1:** [e.g., Weather data]
- **Fundamental Driver 2:** [e.g., Gas prices, renewable generation]

### Data Quality Checks
- Missing value detection and handling
- Outlier identification
- Temporal consistency validation
- Cross-validation with external sources

---

## 🤖 Models

### Baseline Model
- **Type:** [e.g., Linear Regression, ARIMA]
- **Features:** [List key features]
- **Performance:** [Add metrics]

### Improved Model
- **Type:** [e.g., Gradient Boosting, LSTM]
- **Features:** [List key features]
- **Performance:** [Add metrics]

### Validation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

---

## 💹 Trading Application

See [Trading Guidance](docs/trading_guidance.md) for detailed explanation of:
- How forecasts translate to curve positions
- Signal generation methodology
- Risk management considerations
- Conditions for forecast invalidation

---

## 🤖 AI/LLM Integration

**Component:** [Description of AI/LLM usage]

**Purpose:** Automate [specific task] to reduce manual effort

**Implementation:**
- API: [OpenAI/Anthropic/Other]
- Prompts: Logged in `outputs/ai_logs/`
- Output: [Description of generated outputs]

---

## 📈 Results

### Model Performance
| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|----|
| Baseline | TBD | TBD | TBD | TBD |
| Improved | TBD | TBD | TBD | TBD |

### Out-of-Sample Predictions
Available in `submission.csv` (optional)

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
- **Trading Guidance:** See `docs/trading_guidance.md`

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
