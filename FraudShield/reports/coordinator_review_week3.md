# Week 3 — Coordinator Review

**Date**: 2026-04-13  
**Sprint**: Week 3 — XGBoost Integration  
**Reviewer**: Coordinator

---

## Models Trained So Far

| # | Model               | Val AUC-PR | Val F1   | Val MCC  | FP Count | FN Count |
|---|---------------------|-----------|----------|----------|----------|----------|
| 1 | Logistic Regression | 0.9946    | 0.9646   | 0.9305   | 928      | 2,048    |
| 2 | Random Forest       | 0.9995    | 0.9858   | 0.9722   | 56       | 1,141    |
| 3 | XGBoost             | **1.0000**| **0.9996**| **0.9992**| **36**  | **0**    |

---

## Best Model Performance

| Metric       | Value        |
|--------------|-------------|
| **Best AUC-PR** | 1.0000 (XGBoost) |
| **Best F1**     | 0.9996 (XGBoost) |
| **Best MCC**    | 0.9992 (XGBoost) |

---

## Threshold Decision

**Selected threshold strategy**: Max-Recall threshold (0.10)

**Rationale**: Missing fraud costs more than false alarms. In credit card fraud detection,
a false negative (missed fraud) results in direct financial loss to the cardholder and
issuer, while a false positive (legitimate transaction flagged) results only in a temporary
inconvenience that can be resolved via confirmation.

| Threshold          | Precision | Recall | F1     |
|--------------------|----------|--------|--------|
| 0.50 (default)     | 0.9992   | 1.0000 | 0.9996 |
| 0.65 (optimal F1)  | 0.9995   | 1.0000 | 0.9998 |
| 0.10 (max-recall)  | 0.9950   | 1.0000 | 0.9975 |

---

## Error Analysis (XGBoost @ threshold = 0.50)

| Error Type       | Count  | % of Predictions |
|------------------|--------|-----------------|
| False Positives  | 36     | 0.0084%         |
| False Negatives  | 0      | 0.0000%         |

**Observation**: XGBoost achieves **zero false negatives** at the default threshold —
every fraudulent transaction is correctly identified. The 36 false positives represent
a negligible fraction of predictions and are an acceptable trade-off.

---

## Week 4 Update

> **Teacher requested expanded model comparison.**

Action items for Week 4:
- [ ] Add **Decision Tree** classifier
- [ ] Add **LightGBM** classifier
- [ ] Run full **5-model ablation study**
- [ ] Conduct **robustness testing** (balanced vs. imbalanced evaluation)
- [ ] Run **SHAP** analysis on the best model
- [ ] Update experiment log and ablation CSV

---

## Risk Register

| Risk                                        | Mitigation                                  |
|---------------------------------------------|---------------------------------------------|
| Overfitting on balanced validation set       | Robustness test with 95:5 imbalanced split  |
| XGBoost near-perfect AUC-PR may not hold     | Validate on held-out test set               |
| LightGBM dependency may conflict             | Pin version ≥ 4.0 in requirements           |
