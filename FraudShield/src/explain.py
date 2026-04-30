"""
FraudShield -- SHAP Explainability Module
==========================================
Generates SHAP explanations for the best-performing model:
summary bar, beeswarm, waterfall, and force plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import lightgbm as lgb

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocess import preprocess
from src.evaluate import evaluate_model, robustness_test


def generate_shap_explanations(csv_path, best_model, best_model_name):
    """
    Generate SHAP explanations for the best model.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV for preprocessing.
    best_model : estimator
        The best-performing fitted model.
    best_model_name : str
        Name of the best model (e.g. "XGBoost", "LightGBM").
    """
    print("\n" + "=" * 70)
    print(f"  SHAP EXPLAINABILITY -- {best_model_name}")
    print("=" * 70)

    # -- Load data ------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(csv_path)

    # -- Sample 1000 rows from X_test -----------------------------------------
    X_sample = X_test.sample(1000, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    print(f"  Sampled {len(X_sample)} rows from X_test for SHAP analysis")
    print(f"  Fraud in sample: {y_sample.sum()} / {len(y_sample)}")

    # -- Setup paths ----------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_root, "reports", "figures")
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(figures_dir, exist_ok=True)

    # -- Compute SHAP values --------------------------------------------------
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_sample)
    print("  SHAP values computed.")

    # -- Plot 1: Summary Bar --------------------------------------------------
    print("  Generating summary bar plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                      max_display=15)
    plt.title(f"SHAP Feature Importance (Bar) - {best_model_name}", fontsize=14)
    plt.tight_layout()
    bar_path = os.path.join(figures_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {bar_path}")

    # -- Plot 2: Beeswarm -----------------------------------------------------
    print("  Generating beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title(f"SHAP Beeswarm - {best_model_name}", fontsize=14)
    plt.tight_layout()
    beeswarm_path = os.path.join(figures_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {beeswarm_path}")

    # -- Plot 3: Waterfall on one actual fraud transaction ---------------------
    print("  Generating waterfall plot (single fraud example)...")
    fraud_indices = X_sample.index[y_sample == 1]
    if len(fraud_indices) > 0:
        # Use the first fraud transaction in the sample
        fraud_pos = list(X_sample.index).index(fraud_indices[0])
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap_values[fraud_pos], show=False, max_display=15)
        plt.title(f"SHAP Waterfall (Fraud Transaction) - {best_model_name}",
                  fontsize=12)
        plt.tight_layout()
        waterfall_path = os.path.join(figures_dir, "shap_waterfall.png")
        plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {waterfall_path}")
    else:
        print("  [WARN] No fraud transactions in sample for waterfall plot")

    # -- Plot 4: Force plot (HTML) --------------------------------------------
    print("  Generating force plot (HTML)...")
    shap.initjs()
    # Use first 100 samples for the force plot
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values.values[:100],
        X_sample.iloc[:100],
        show=False,
    )
    force_path = os.path.join(figures_dir, "shap_force_plot.html")
    shap.save_html(force_path, force_html)
    print(f"  [SAVED] {force_path}")

    # -- Top 5 features by mean |SHAP| ----------------------------------------
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_sample.columns)
    top5 = feature_importance.nlargest(5)

    print("\n  Top 5 Features by Mean |SHAP|:")
    print("  " + "-" * 40)
    for feat, val in top5.items():
        print(f"    {feat:20s}  {val:.6f}")
    print("  " + "-" * 40)

    # -- Write SHAP limitation note -------------------------------------------
    limitation_path = os.path.join(reports_dir, "shap_limitation.txt")
    with open(limitation_path, "w") as f:
        f.write(
            "V1-V28 are PCA-anonymized. SHAP identifies which features drive "
            "predictions but cannot provide business-level explanation. Amount "
            "and log1p_amount are the only human-interpretable features.\n"
        )
    print(f"  [SAVED] {limitation_path}")

    print("\n" + "=" * 70)
    print("  SHAP analysis complete.")
    print("=" * 70)

    return shap_values, top5


# -- CLI entry point ----------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # -- 1. Load data ---------------------------------------------------------
    csv_file = os.path.join(project_root, "data", "creditcard_2023.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(csv_file)

    # -- 2. Load all 5 saved models -------------------------------------------
    models_dir = os.path.join(project_root, "models")

    lr_model = joblib.load(os.path.join(models_dir, "lr_model.pkl"))
    dt_model = joblib.load(os.path.join(models_dir, "dt_model.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(models_dir, "xgb_model.json"))

    lgbm_model = joblib.load(os.path.join(models_dir, "lgbm_model.pkl"))

    model_dict = {
        "LogisticRegression": lr_model,
        "DecisionTree": dt_model,
        "RandomForest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
    }

    # -- 3. Full ablation to pick best model ----------------------------------
    all_metrics = []
    for name, model in model_dict.items():
        m = evaluate_model(model, X_val, y_val, name)
        all_metrics.append(m)

    comparison_df = pd.DataFrame(all_metrics).set_index("model")
    comparison_df = comparison_df.sort_values("auc_pr", ascending=False)
    best_model_name = comparison_df.index[0]
    best_model = model_dict[best_model_name]

    print("\n" + "=" * 80)
    print(f"  [RESULTS] Full 5-Model Ablation Table (Validation Set)")
    print("=" * 80)
    print(comparison_df.to_string())
    print("=" * 80)
    print(f"\n  >>> BEST MODEL: {best_model_name} "
          f"(AUC-PR = {comparison_df.iloc[0]['auc_pr']:.4f})")

    # -- 4. Robustness test ---------------------------------------------------
    robustness_test(csv_file, best_model, best_model_name)

    # -- 5. SHAP explanations -------------------------------------------------
    generate_shap_explanations(csv_file, best_model, best_model_name)
