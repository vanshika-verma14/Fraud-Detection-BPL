# 🛡️ FraudShield

**Credit Card Fraud Detection using Machine Learning**

A 6-week academic PBL project that builds an end-to-end ML pipeline for detecting fraudulent credit card transactions.

> 🔗 **Live Demo**: *[fill after deploy]*

---

## Project Overview

FraudShield is an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The project covers the full lifecycle — from exploratory data analysis and feature engineering through model training, evaluation, explainability (SHAP), and deployment via a Streamlit dashboard.

Key highlights:
- **Preprocessing**: log-transform + StandardScaler on Amount; PCA features (V1–V28) left as-is
- **Modeling**: 5-model ablation — Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- **Threshold Tuning**: Full sweep with optimal F1 and max-recall thresholds
- **Error Analysis**: FP/FN breakdown with feature attribution
- **Explainability**: SHAP values for global & local feature importance
- **Robustness**: Performance tested under balanced vs imbalanced class distributions
- **App**: Interactive 3-page Streamlit dashboard for batch & single-transaction predictions

---

## Dataset

We use the **Credit Card Fraud Detection 2023** dataset from Kaggle.

| Property | Value |
|----------|-------|
| Rows | 568,630 |
| Features | `V1`–`V28` (PCA), `Amount` |
| Target | `Class` (0 = legit, 1 = fraud) |
| Balance | ~50/50 |
| Nulls | None |

> ⚠️ The dataset is **~325 MB** and is **not included** in this repository. See the setup instructions below to download it automatically.

---

## Quick Start (3 Steps)

```bash
# 1. Clone and install
git clone <repo-url>
cd FraudShield
pip install -r requirements.txt

# 2. Download dataset from Kaggle (automatic)
python download_data.py

# 3. Run end-to-end pipeline
bash run_pipeline.sh        # Linux / macOS
run_pipeline.bat            # Windows
```

Then launch the interactive dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

## Installation (Detailed)

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd FraudShield
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

```bash
python download_data.py
```

This uses [`kagglehub`](https://github.com/Kaggle/kagglehub) to download the dataset automatically from Kaggle and places it in `data/creditcard_2023.csv`.

**If you get an authentication error**, do one of these:

**Option A** — Set environment variables:
```bash
# Windows
set KAGGLE_USERNAME=your_kaggle_username
set KAGGLE_KEY=your_kaggle_api_key

# Linux/Mac
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```

**Option B** — Create a `kaggle.json` file:
1. Go to [kaggle.com](https://www.kaggle.com) → Your Profile → Account → **Create New API Token**
2. Save the downloaded `kaggle.json` to:
   - Windows: `C:\Users\<YourName>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

**Option C** — Manual download:
1. Go to [Kaggle — Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
2. Click **Download**, extract, and copy `creditcard_2023.csv` into the `data/` folder.

---

## How to Run

### End-to-End Pipeline (recommended)

```bash
bash run_pipeline.sh        # Linux / macOS
run_pipeline.bat            # Windows
```

This runs all 4 stages:
1. **Preprocessing** — `python src/preprocess.py`
2. **Training** — `python src/train.py` (LR, DT, RF, XGBoost, LightGBM)
3. **Evaluation** — `python src/evaluate.py` (ablation, robustness test)
4. **Explainability** — `python src/explain.py` (SHAP analysis)

### Individual Steps

```bash
python download_data.py                        # Download dataset
python src/preprocess.py                       # Preprocessing only
python src/train.py                            # Train all 5 models
python src/evaluate.py                         # Evaluate + ablation + robustness
python src/explain.py                          # SHAP explainability
```

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Results

### 5-Model Comparison (Validation Set)

| Model | Accuracy | Precision | Recall | F1 | AUC-PR | MCC | FP | FN |
|-------|----------|-----------|--------|------|--------|------|-----|-----|
| **XGBoost** | **99.96%** | **99.92%** | **100%** | **0.9996** | **1.0000** | **0.9992** | **36** | **0** |
| LightGBM | 99.93% | 99.87% | 100% | 0.9993 | 1.0000 | 0.9986 | 57 | 1 |
| Random Forest | 98.60% | 99.87% | 97.32% | 0.9858 | 0.9995 | 0.9722 | 56 | 1141 |
| Decision Tree | 98.58% | 98.32% | 98.84% | 0.9858 | 0.9960 | 0.9715 | 722 | 493 |
| Logistic Regression | 96.51% | 97.77% | 95.20% | 0.9646 | 0.9946 | 0.9305 | 928 | 2048 |

> 🏆 **Best Model: XGBoost** — 0 false negatives, AUC-PR = 1.0000

### Threshold Analysis
- **Optimal F1 threshold**: 0.65
- **Max-Recall threshold**: 0.10 (216 FPs, 0 FNs)

### Robustness
The best model shows near-zero degradation when moving from a balanced (50:50) to highly imbalanced (95:5) test set. Precision, Recall, F1, and MCC all remain stable above 0.998.

---

## Limitations

1. **PCA Anonymization**: Features V1–V28 are PCA-transformed and anonymized. While SHAP can identify which mathematical components drive predictions, business-level explanations (e.g., "flagged due to unusual purchase location") are not possible. Only **Amount** is human-interpretable.
2. **Balanced Dataset**: The training data is perfectly balanced (50/50), which may not reflect real-world fraud rates (~0.1–0.5%). Robustness tests show the model handles imbalance well, but production deployment should include recalibration.
3. **No Temporal Features**: Transaction timestamps were removed during PCA. Real-world fraud detection benefits from temporal patterns (time-of-day, transaction frequency, velocity checks).
4. **Static Model**: The pipeline trains offline. Production systems need continuous learning to adapt to evolving fraud patterns.
5. **Single Dataset**: Results are validated on one dataset. Cross-dataset generalization is untested.

---

## Project Structure

```
FraudShield/
├── app/
│   └── streamlit_app.py           # 3-page Streamlit dashboard
├── data/                          # Dataset (.csv git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_baseline.ipynb          # LR & RF baseline comparison
│   ├── 03_experiments.ipynb       # XGBoost, ablation, threshold sweep
│   ├── 04_explainability.ipynb    # SHAP, robustness, 5-model final
│   └── run_03_experiments.py      # Standalone script for generating plots
├── src/
│   ├── preprocess.py              # Data preprocessing & splitting
│   ├── train.py                   # Model training (5 models)
│   ├── evaluate.py                # Evaluation, ablation, robustness
│   └── explain.py                 # SHAP explainability
├── models/                        # Saved models (git-ignored)
├── reports/
│   ├── figures/                   # Generated plots & CSVs
│   ├── experiment_log.csv         # Full experiment tracking
│   └── demo_readiness_checklist.md
├── tests/
│   └── test_pipeline.py           # Pytest test suite
├── download_data.py               # ⬇️ One-command dataset downloader
├── requirements.txt               # Python dependencies
├── run_pipeline.sh                # One-command pipeline runner (Linux/macOS)
├── run_pipeline.bat               # One-command pipeline runner (Windows)
└── README.md
```

---

## Tech Stack

| Category       | Tools                                    |
|----------------|------------------------------------------|
| ML / Stats     | scikit-learn, XGBoost, LightGBM, SciPy   |
| Explainability | SHAP                                     |
| Visualization  | Matplotlib, Seaborn                      |
| Data Download  | kagglehub                                |
| App            | Streamlit                                |
| Testing        | pytest                                   |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Data exploration — nulls, class distribution, feature correlations |
| `02_baseline.ipynb` | Week 2 — Logistic Regression & Random Forest baseline |
| `03_experiments.ipynb` | Week 3 — XGBoost champion, ablation table, threshold sweep, PR curves, error analysis |
| `04_explainability.ipynb` | Week 4 — 5-model ablation, robustness, SHAP explainability, final summary |

To view notebooks: `python -m jupyter notebook` and navigate to `notebooks/`.

---

## Team

| Name | Role |
|------|------|
| *Add team member* | *Role* |
| *Add team member* | *Role* |
| *Add team member* | *Role* |

---

## License

This project is for educational and research purposes.
