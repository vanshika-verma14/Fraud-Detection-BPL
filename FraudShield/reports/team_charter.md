# FraudShield — Team Charter

| Field            | Details                                                                      |
|------------------|-----------------------------------------------------------------------------|
| **Project**      | Credit Card Fraud Detection                                                  |
| **Team**         | BlazeBuilders                                                               |
| **Track**        | D                                                                           |
| **Problem**      | Detect fraudulent transactions using ML on 568K-row European cardholder dataset |
| **Success Metric** | AUC-PR >= 0.95 on validation set                                          |
| **Dataset**      | `nelgiriyewithana/credit-card-fraud-detection-dataset-2023`                  |

---

## Roles

| Role              | Responsibility                                                        |
|-------------------|----------------------------------------------------------------------|
| **Coordinator**   | Sprint planning, milestone tracking, teacher communication            |
| **Data Lead**     | EDA, feature engineering, preprocessing pipeline                      |
| **ML Lead**       | Model training, ablation study, hyperparameter tuning, SHAP analysis  |
| **SWE Lead**      | Code architecture, CI/CD, Streamlit dashboard                         |
| **QA-Doc Lead**   | Testing, documentation, experiment logging, report generation         |

---

## Milestones

| Week | Milestone          | Deliverables                                              |
|------|--------------------|----------------------------------------------------------|
| W1   | **Setup**          | Repo init, data download, directory structure, EDA        |
| W2   | **Baseline**       | LR + RF trained, evaluate pipeline, ablation table        |
| W3   | **XGBoost**        | XGBoost champion, threshold sweep, error analysis         |
| W4   | **All5Models+SHAP**| DT + LightGBM added, 5-model ablation, robustness, SHAP  |
| W5   | **Demo**           | Streamlit dashboard, live demo ready                      |
| W6   | **Report**         | Final report, presentation, GitHub cleanup                |

---

## Communication

- **Daily**: Async updates in team channel
- **Weekly**: Sync meeting with teacher — present progress against milestones
- **Code Review**: All PRs require at least one peer review before merge
