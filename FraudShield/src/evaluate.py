"""
FraudShield -- Model Evaluation Module
======================================
Evaluates trained classifiers on the validation set and produces a
side-by-side comparison table, threshold sweep, error analysis,
and robustness testing.
"""

import os
import sys
import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocess import preprocess


def evaluate_model(model, X, y, model_name: str, threshold: float = 0.5) -> dict:
    """
    Evaluate a binary classifier and return a metrics dictionary.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model with ``.predict()`` and ``.predict_proba()`` methods.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True labels (0 / 1).
    model_name : str
        Human-readable name for console output.
    threshold : float
        Decision threshold for converting probabilities to predictions.

    Returns
    -------
    dict
        Keys: model, accuracy, precision, recall, f1, roc_auc,
              auc_pr, mcc, fp_count, fn_count.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="binary")
    rec = recall_score(y, y_pred, average="binary")
    f1 = f1_score(y, y_pred, average="binary")
    roc = roc_auc_score(y, y_proba)
    auc_pr = average_precision_score(y, y_proba)
    mcc = matthews_corrcoef(y, y_pred)

    # Confusion-matrix breakdown
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # -- Pretty-print ---------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  {model_name}")
    print(f"{'=' * 55}")
    print(classification_report(y, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    print(cm)
    print(f"\n  ROC-AUC   : {roc:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  FP count  : {fp}")
    print(f"  FN count  : {fn}")
    print(f"{'=' * 55}\n")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4),
        "auc_pr": round(auc_pr, 4),
        "mcc": round(mcc, 4),
        "fp_count": int(fp),
        "fn_count": int(fn),
    }


# ---------------------------------------------------------------------------
#  Threshold sweep
# ---------------------------------------------------------------------------
def threshold_sweep(model, X, y):
    """
    Sweep decision thresholds from 0.1 to 0.9 and compute precision,
    recall, and F1 at each point.

    Returns
    -------
    sweep_df : pd.DataFrame
        Columns: threshold, precision, recall, f1
    optimal_f1_threshold : float
        Threshold that maximises F1.
    max_recall_threshold : float
        Threshold that maximises recall among those where precision > 0.5.
    """
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        p = precision_score(y, y_pred_t, average="binary", zero_division=0)
        r = recall_score(y, y_pred_t, average="binary", zero_division=0)
        f = f1_score(y, y_pred_t, average="binary", zero_division=0)
        rows.append({"threshold": round(t, 2), "precision": p, "recall": r, "f1": f})

    sweep_df = pd.DataFrame(rows)

    # Best F1
    best_idx = sweep_df["f1"].idxmax()
    optimal_f1_threshold = sweep_df.loc[best_idx, "threshold"]

    # Max recall where precision > 0.5
    valid = sweep_df[sweep_df["precision"] > 0.5]
    if len(valid) > 0:
        max_recall_idx = valid["recall"].idxmax()
        max_recall_threshold = valid.loc[max_recall_idx, "threshold"]
    else:
        max_recall_threshold = 0.5  # fallback

    return sweep_df, optimal_f1_threshold, max_recall_threshold


# ---------------------------------------------------------------------------
#  Error analysis
# ---------------------------------------------------------------------------
def error_analysis(model, X, y, threshold, feature_names):
    """
    Classify errors at a given threshold and print diagnostic info.

    Returns
    -------
    fp_count : int
    fn_count : int
    """
    y_true = np.asarray(y)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    total = len(y_true)

    print(f"\n{'=' * 60}")
    print(f"  Error Analysis  (threshold = {threshold})")
    print(f"{'=' * 60}")
    print(f"  False Positives : {fp_count}  ({fp_count / total:.4%} of predictions)")
    print(f"  False Negatives : {fn_count}  ({fn_count / total:.4%} of predictions)")

    # Convert to DataFrame for slicing
    X_df = pd.DataFrame(X, columns=feature_names) if not isinstance(X, pd.DataFrame) else X.copy()
    X_df = X_df.reset_index(drop=True)

    v_cols = [c for c in feature_names if c.startswith("V")]

    # -- False Positives examples ---------------------------------------------
    fp_indices = np.where(fp_mask)[0]
    if len(fp_indices) > 0:
        print(f"\n  -- Top 5 False Positives --")
        for i, idx in enumerate(fp_indices[:5]):
            row = X_df.iloc[idx]
            top_v = row[v_cols].abs().nlargest(3)
            top_v_str = ", ".join(f"{k}={v:.3f}" for k, v in top_v.items())
            amount_val = row.get("Amount", row.get("log1p_amount", "N/A"))
            print(f"    FP-{i+1}: Amount={amount_val:.4f}  | top V: {top_v_str}")
    else:
        print("  No False Positives at this threshold.")

    # -- False Negatives examples ---------------------------------------------
    fn_indices = np.where(fn_mask)[0]
    if len(fn_indices) > 0:
        print(f"\n  -- Top 5 False Negatives --")
        for i, idx in enumerate(fn_indices[:5]):
            row = X_df.iloc[idx]
            top_v = row[v_cols].abs().nlargest(3)
            top_v_str = ", ".join(f"{k}={v:.3f}" for k, v in top_v.items())
            amount_val = row.get("Amount", row.get("log1p_amount", "N/A"))
            print(f"    FN-{i+1}: Amount={amount_val:.4f}  | top V: {top_v_str}")
    else:
        print("  No False Negatives at this threshold.")

    print(f"{'=' * 60}\n")

    return fp_count, fn_count


# ---------------------------------------------------------------------------
#  Robustness test
# ---------------------------------------------------------------------------
def robustness_test(csv_path, best_model, best_model_name):
    """
    Robustness test: evaluate best model under balanced and imbalanced
    conditions, then attempt recovery with class-weight adjustment.

    Steps
    -----
    A. Evaluate on balanced X_test / y_test at threshold=0.5
    B. Construct 95:5 imbalanced test set (keep all fraud, sample legit 19:1)
    C. Evaluate on imbalanced set at threshold=0.5 — record precision drop
    D. Retrain with class-weight adjustment and evaluate on same imbalanced set

    Parameters
    ----------
    csv_path : str
        Path to raw CSV for preprocessing.
    best_model : estimator
        The best-performing fitted model.
    best_model_name : str
        Name of the best model (e.g. "XGBoost", "LightGBM").
    """
    print("\n" + "=" * 70)
    print("  ROBUSTNESS TEST")
    print("=" * 70)

    # -- Reload data ----------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(csv_path)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    log_path = os.path.join(project_root, "reports", "experiment_log.csv")

    # ── Step A: Balanced test set ────────────────────────────────────
    print("\n  [Step A] Balanced test set (threshold=0.5)")
    bal = evaluate_model(best_model, X_test, y_test, f"{best_model_name} (balanced)", threshold=0.5)

    _append_log(log_path, best_model_name, bal,
                "robustness: balanced test set evaluation")

    # ── Step B: Construct 95:5 imbalanced test set ───────────────────
    print("\n  [Step B] Constructing 95:5 imbalanced test set...")
    fraud_mask = y_test == 1
    legit_mask = y_test == 0

    X_fraud = X_test[fraud_mask]
    y_fraud = y_test[fraud_mask]

    n_fraud = len(y_fraud)
    n_legit_target = n_fraud * 19  # 95:5 ratio → 19 legit per 1 fraud

    X_legit = X_test[legit_mask]
    y_legit = y_test[legit_mask]

    if n_legit_target <= len(y_legit):
        sample_idx = np.random.RandomState(42).choice(
            len(y_legit), size=n_legit_target, replace=False
        )
        X_legit_sample = X_legit.iloc[sample_idx]
        y_legit_sample = y_legit.iloc[sample_idx]
    else:
        X_legit_sample = X_legit
        y_legit_sample = y_legit

    X_imb = pd.concat([X_fraud, X_legit_sample], axis=0).reset_index(drop=True)
    y_imb = pd.concat([y_fraud, y_legit_sample], axis=0).reset_index(drop=True)

    fraud_ratio = y_imb.mean()
    print(f"    Imbalanced set: {len(y_imb)} rows, fraud ratio = {fraud_ratio:.4f}")

    # ── Step C: Evaluate on imbalanced set ───────────────────────────
    print("\n  [Step C] Evaluating on imbalanced set (threshold=0.5)")
    imb = evaluate_model(best_model, X_imb, y_imb, f"{best_model_name} (imbalanced)", threshold=0.5)

    _append_log(log_path, best_model_name, imb,
                "robustness: 95:5 imbalanced test set")

    # ── Step D: Retrain with class-weight, evaluate again ────────────
    print(f"\n  [Step D] Retraining {best_model_name} with class-weight adjustment...")

    if best_model_name == "XGBoost":
        recovered_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="aucpr",
            scale_pos_weight=19,
            early_stopping_rounds=20,
        )
        recovered_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif best_model_name == "LightGBM":
        recovered_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        recovered_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
    else:
        # Generic fallback: retrain with class_weight if supported
        from sklearn.base import clone
        recovered_model = clone(best_model)
        if hasattr(recovered_model, "class_weight"):
            recovered_model.set_params(class_weight="balanced")
        recovered_model.fit(X_train, y_train)

    rec = evaluate_model(recovered_model, X_imb, y_imb,
                         f"{best_model_name} (recovered)", threshold=0.5)

    _append_log(log_path, best_model_name, rec,
                "robustness: class-weight adjusted, imbalanced test set")

    # ── Side-by-side comparison ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ROBUSTNESS COMPARISON -- Balanced | Imbalanced | Recovered")
    print("=" * 70)
    header = f"  {'Metric':<12s}  {'Balanced':>10s}  {'Imbalanced':>10s}  {'Recovered':>10s}"
    print(header)
    print("  " + "-" * 48)
    for metric in ["precision", "recall", "f1", "mcc"]:
        print(f"  {metric:<12s}  {bal[metric]:>10.4f}  {imb[metric]:>10.4f}  {rec[metric]:>10.4f}")
    print("=" * 70)

    # ── Save robustness comparison chart ─────────────────────────────
    metrics_list = ["precision", "recall", "f1", "mcc"]
    bal_vals = [bal[m] for m in metrics_list]
    imb_vals = [imb[m] for m in metrics_list]
    rec_vals = [rec[m] for m in metrics_list]

    x = np.arange(len(metrics_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, bal_vals, width, label="Balanced", color="#2ecc71", edgecolor="white")
    bars2 = ax.bar(x, imb_vals, width, label="Imbalanced", color="#e74c3c", edgecolor="white")
    bars3 = ax.bar(x + width, rec_vals, width, label="Recovered", color="#3498db", edgecolor="white")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Robustness Test - {best_model_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_list])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    chart_path = os.path.join(figures_dir, "robustness_comparison.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\n  [SAVED] Robustness chart -> {chart_path}")


def _append_log(log_path, model_name, metrics, notes):
    """Append a row to the experiment log CSV."""
    from datetime import date
    row = {
        "date": date.today().isoformat(),
        "model": model_name,
        "hyperparams": "",
        "train_f1": "",
        "val_f1": metrics.get("f1", ""),
        "val_auc_pr": metrics.get("auc_pr", ""),
        "val_recall": metrics.get("recall", ""),
        "val_precision": metrics.get("precision", ""),
        "val_mcc": metrics.get("mcc", ""),
        "val_roc_auc": metrics.get("roc_auc", ""),
        "val_accuracy": metrics.get("accuracy", ""),
        "fp_count": metrics.get("fp_count", ""),
        "fn_count": metrics.get("fn_count", ""),
        "threshold": 0.5,
        "notes": notes,
    }
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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

    # -- 3. Evaluate all 5 on validation set ----------------------------------
    all_metrics = []
    for name, model in model_dict.items():
        m = evaluate_model(model, X_val, y_val, name)
        all_metrics.append(m)

    # -- 4. Full 5-model ablation table ---------------------------------------
    comparison_df = pd.DataFrame(all_metrics).set_index("model")

    # Sort by AUC-PR descending
    comparison_df = comparison_df.sort_values("auc_pr", ascending=False)

    best_model_name = comparison_df.index[0]
    print("\n" + "=" * 80)
    print(f"  [RESULTS] Full 5-Model Ablation Table (Validation Set)")
    print("=" * 80)
    print(comparison_df.to_string())
    print("=" * 80)
    print(f"\n  >>> BEST MODEL: {best_model_name} (AUC-PR = {comparison_df.iloc[0]['auc_pr']:.4f})")

    # Save ablation
    figures_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    ablation_path = os.path.join(figures_dir, "ablation_5models.csv")
    comparison_df.to_csv(ablation_path)
    print(f"  [SAVED] Ablation table -> {ablation_path}")

    # -- 4b. Model progression bar chart --------------------------------------
    model_order = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
    aucpr_vals = [comparison_df.loc[m, "auc_pr"] for m in model_order]
    colors = ["#95a5a6", "#95a5a6", "#f39c12", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(model_order, aucpr_vals, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, aucpr_vals):
        ax.text(bar.get_width() - 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="right", fontsize=11,
                color="white", fontweight="bold")
    ax.set_xlabel("AUC-PR (Validation Set)", fontsize=12)
    ax.set_title("Model Progression: Finding the Best Fraud Detector", fontsize=14, fontweight="bold")
    ax.set_xlim(0.99, 1.001)
    ax.axvline(x=0.95, color="red", linestyle="--", alpha=0.5, label="Target AUC-PR >= 0.95")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    progression_path = os.path.join(figures_dir, "model_progression.png")
    fig.savefig(progression_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Model progression chart -> {progression_path}")

    # -- 5. Threshold sweep on best model -------------------------------------
    best_model = model_dict[best_model_name]
    print(f"\n--- Threshold Sweep ({best_model_name}) ---")
    sweep_df, best_f1_t, max_rec_t = threshold_sweep(best_model, X_val, y_val)
    print(sweep_df.to_string(index=False))
    print(f"\n  Optimal F1 threshold   : {best_f1_t}")
    print(f"  Max-Recall threshold   : {max_rec_t}")

    # -- 6. Error analysis on best model at max-Recall threshold ---------------
    error_analysis(best_model, X_val, y_val, max_rec_t, list(X_val.columns))

    # -- 7. Robustness test ---------------------------------------------------
    robustness_test(csv_file, best_model, best_model_name)
