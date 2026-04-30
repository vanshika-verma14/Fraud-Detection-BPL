#!/bin/bash
set -e
echo "=== FraudShield Pipeline ==="
echo "[1/4] Preprocessing..."
python src/preprocess.py
echo "[2/4] Training all 5 models..."
python src/train.py
echo "[3/4] Evaluating, comparing, robustness test..."
python src/evaluate.py
echo "[4/4] SHAP explainability on best model..."
python src/explain.py
echo ""
echo "=== Done. Launch demo: streamlit run app/streamlit_app.py ==="
