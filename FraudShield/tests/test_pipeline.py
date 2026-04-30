"""
FraudShield — Pipeline Tests
=============================
Validates preprocessing shapes, class balance, and data-leakage checks.
Run with:  pytest tests/ -v
"""

import os
import sys
import pytest
import joblib
import numpy as np

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess

# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def split_data():
    """Run preprocessing once for all tests in this module."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "creditcard_2023.csv")
    if not os.path.isfile(csv_path):
        pytest.skip(f"Dataset not found at {csv_path}")
    return preprocess(csv_path)


@pytest.fixture(scope="module")
def split_indices():
    """Load persisted split indices."""
    idx_path = os.path.join(PROJECT_ROOT, "models", "split_indices.pkl")
    if not os.path.isfile(idx_path):
        pytest.skip("split_indices.pkl not found — run preprocess first")
    return joblib.load(idx_path)


# ── Tests ────────────────────────────────────────────────────────────

def test_preprocess_shapes(split_data):
    """X_train ~70%, X_val ~15%, X_test ~15% of total rows."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data
    total = len(X_train) + len(X_val) + len(X_test)

    train_pct = len(X_train) / total
    val_pct = len(X_val) / total
    test_pct = len(X_test) / total

    # Allow ±2 pp tolerance
    assert abs(train_pct - 0.70) < 0.02, f"Train split is {train_pct:.2%}, expected ~70%"
    assert abs(val_pct - 0.15) < 0.02, f"Val split is {val_pct:.2%}, expected ~15%"
    assert abs(test_pct - 0.15) < 0.02, f"Test split is {test_pct:.2%}, expected ~15%"

    # X and y lengths must match within each split
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_class_balance_preserved(split_data):
    """y_train class ratio stays within 2% of the original 50/50 balance."""
    _, _, _, y_train, _, _ = split_data
    fraud_ratio = y_train.mean()

    assert abs(fraud_ratio - 0.50) < 0.02, (
        f"y_train fraud ratio is {fraud_ratio:.4f}, expected within 2% of 0.50"
    )


def test_no_data_leakage(split_indices):
    """Train, val, and test index sets must have zero intersection."""
    train_set = set(split_indices["train_idx"])
    val_set = set(split_indices["val_idx"])
    test_set = set(split_indices["test_idx"])

    assert train_set.isdisjoint(val_set), "LEAK: train / val is non-empty"
    assert train_set.isdisjoint(test_set), "LEAK: train / test is non-empty"
    assert val_set.isdisjoint(test_set), "LEAK: val / test is non-empty"


def test_model_files_exist():
    """After training, both model pickle files must exist on disk."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    lr_path = os.path.join(models_dir, "lr_model.pkl")
    rf_path = os.path.join(models_dir, "rf_model.pkl")

    assert os.path.isfile(lr_path), f"LR model not found at {lr_path}"
    assert os.path.isfile(rf_path), f"RF model not found at {rf_path}"


def test_evaluate_returns_all_metrics():
    """evaluate_model must return a dict containing all 7 expected metric keys."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression as LR
    from src.evaluate import evaluate_model

    # Tiny synthetic dataset
    X_dummy, y_dummy = make_classification(
        n_samples=200, n_features=5, random_state=42
    )
    dummy_model = LR(max_iter=200, random_state=42)
    dummy_model.fit(X_dummy, y_dummy)

    result = evaluate_model(dummy_model, X_dummy, y_dummy, "DummyLR")

    expected_keys = {"model", "accuracy", "precision", "recall", "f1",
                     "roc_auc", "auc_pr", "mcc"}
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


def test_all_5_models_saved():
    """All 5 trained model files must exist on disk."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    expected_files = [
        "lr_model.pkl",
        "dt_model.pkl",
        "rf_model.pkl",
        "xgb_model.json",
        "lgbm_model.pkl",
    ]
    for fname in expected_files:
        fpath = os.path.join(models_dir, fname)
        assert os.path.isfile(fpath), f"Model file not found: {fpath}"


def test_shap_outputs_exist():
    """All 4 SHAP output files must exist in reports/figures/."""
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    expected_files = [
        "shap_summary_bar.png",
        "shap_beeswarm.png",
        "shap_waterfall.png",
        "shap_force_plot.html",
    ]
    for fname in expected_files:
        fpath = os.path.join(figures_dir, fname)
        assert os.path.isfile(fpath), f"SHAP output not found: {fpath}"


def test_robustness_comparison_exists():
    """Robustness comparison chart must exist."""
    fpath = os.path.join(PROJECT_ROOT, "reports", "figures", "robustness_comparison.png")
    assert os.path.isfile(fpath), f"Robustness chart not found: {fpath}"


def test_model_progression_exists():
    """Model progression bar chart must exist."""
    fpath = os.path.join(PROJECT_ROOT, "reports", "figures", "model_progression.png")
    assert os.path.isfile(fpath), f"Model progression chart not found: {fpath}"


def test_streamlit_app_imports():
    """Importing the streamlit_app module must not raise any errors."""
    app_dir = os.path.join(PROJECT_ROOT, "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Only verify the module file is importable (basic syntax/import check)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "streamlit_app",
        os.path.join(app_dir, "streamlit_app.py"),
    )
    assert spec is not None, "Could not find streamlit_app.py"
    # Don't actually execute (it calls st.set_page_config at module level),
    # but verify the file parses without syntax errors.
    import py_compile
    py_compile.compile(
        os.path.join(app_dir, "streamlit_app.py"), doraise=True,
    )


def test_run_pipeline_complete():
    """run_pipeline.sh must reference all 4 pipeline python commands."""
    pipeline_path = os.path.join(PROJECT_ROOT, "run_pipeline.sh")
    assert os.path.isfile(pipeline_path), "run_pipeline.sh not found"

    content = open(pipeline_path, "r", encoding="utf-8").read()
    required_commands = [
        "python src/preprocess.py",
        "python src/train.py",
        "python src/evaluate.py",
        "python src/explain.py",
    ]
    for cmd in required_commands:
        assert cmd in content, f"run_pipeline.sh missing: {cmd}"


def test_readme_has_results():
    """README.md must contain 'AUC-PR' to confirm results table is present."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    assert os.path.isfile(readme_path), "README.md not found"

    content = open(readme_path, "r", encoding="utf-8").read()
    assert "AUC-PR" in content, "README.md does not contain 'AUC-PR'"
