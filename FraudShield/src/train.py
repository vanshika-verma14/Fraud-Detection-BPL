"""
FraudShield -- Model Training Module
====================================
Trains Logistic Regression, Decision Tree, Random Forest, XGBoost, and
LightGBM classifiers on the preprocessed credit-card fraud dataset and
persists them to disk.
"""

import os
import sys
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.preprocess import preprocess


def train_models(csv_path: str):
    """
    Train LR, DT, RF, XGBoost, and LightGBM on the fraud dataset.

    Parameters
    ----------
    csv_path : str
        Path to the raw ``creditcard_2023.csv`` file.

    Returns
    -------
    dict
        Mapping of model name → (fitted model, training time in seconds).
    """
    # -- 1. Preprocess --------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(csv_path)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    results = {}

    # -- 2. Logistic Regression -----------------------------------------------
    print("Training Logistic Regression...")
    t0 = time.time()
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - t0
    joblib.dump(lr_model, os.path.join(models_dir, "lr_model.pkl"))
    print(f"  [OK] Logistic Regression trained. ({lr_time:.2f}s)")
    results["LogisticRegression"] = (lr_model, lr_time)

    # -- 3. Decision Tree -----------------------------------------------------
    print("Training Decision Tree...")
    t0 = time.time()
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
    )
    dt_model.fit(X_train, y_train)
    dt_time = time.time() - t0
    joblib.dump(dt_model, os.path.join(models_dir, "dt_model.pkl"))
    print(f"  [OK] Decision Tree trained. ({dt_time:.2f}s)")
    results["DecisionTree"] = (dt_model, dt_time)

    # -- 4. Random Forest -----------------------------------------------------
    print("Training Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    print(f"  [OK] Random Forest trained. ({rf_time:.2f}s)")
    results["RandomForest"] = (rf_model, rf_time)

    # -- 5. XGBoost -----------------------------------------------------------
    print("Training XGBoost...")
    t0 = time.time()
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="aucpr",
        early_stopping_rounds=20,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_time = time.time() - t0
    xgb_model.save_model(os.path.join(models_dir, "xgb_model.json"))
    print(f"  [OK] XGBoost trained. ({xgb_time:.2f}s)")
    results["XGBoost"] = (xgb_model, xgb_time)

    # -- 6. LightGBM ----------------------------------------------------------
    print("Training LightGBM...")
    t0 = time.time()
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )
    lgbm_time = time.time() - t0
    joblib.dump(lgbm_model, os.path.join(models_dir, "lgbm_model.pkl"))
    print(f"  [OK] LightGBM trained. ({lgbm_time:.2f}s)")
    results["LightGBM"] = (lgbm_model, lgbm_time)

    return results


# -- CLI entry point ----------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(project_root, "data", "creditcard_2023.csv")

    results = train_models(csv_file)

    print("\n" + "=" * 55)
    print("  Training Summary")
    print("=" * 55)
    for name, (model, t) in results.items():
        print(f"  {name:25s}  {t:8.2f}s")
    print("=" * 55)
    print("\nAll 5 models saved.")
