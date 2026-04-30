"""
FraudShield — Data Preprocessing Module
========================================
Loads raw CSV, engineers features, scales selected columns,
and produces stratified train / val / test splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess(csv_path: str):
    """
    End-to-end preprocessing for the credit-card fraud dataset.

    Steps
    -----
    1. Drop the ``id`` column.
    2. Create ``log1p_amount = np.log1p(Amount)``.
    3. StandardScaler on ``['Amount', 'log1p_amount']`` only (V1–V28 are
       already PCA-transformed).
    4. Stratified 70 / 15 / 15 train / val / test split.
    5. Persist split indices to ``models/split_indices.pkl``.

    Parameters
    ----------
    csv_path : str
        Path to the raw ``creditcard_2023.csv`` file.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : pd.DataFrame / pd.Series
    """
    # ── 1. Load & drop id ────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # ── 2. Feature engineering ───────────────────────────────────────
    df["log1p_amount"] = np.log1p(df["Amount"])

    # ── 3. Scale only Amount columns ─────────────────────────────────
    scale_cols = ["Amount", "log1p_amount"]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Save scaler for later inference use
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # ── 4. Separate features / target ────────────────────────────────
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ── 5. Stratified 70 / 15 / 15 split ────────────────────────────
    #   First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    #   Second split: 50% of temp → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    # ── 6. Persist split indices ─────────────────────────────────────
    split_indices = {
        "train_idx": X_train.index.tolist(),
        "val_idx": X_val.index.tolist(),
        "test_idx": X_test.index.tolist(),
    }
    joblib.dump(split_indices, os.path.join(models_dir, "split_indices.pkl"))

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve path relative to project root (FraudShield/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(project_root, "data", "creditcard_2023.csv")

    print(f"Loading dataset from: {csv_file}")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(csv_file)

    total = len(X_train) + len(X_val) + len(X_test)
    print("\n-- Split Shapes ------------------------------------")
    print(f"  Total rows : {total:,}")
    print(f"  X_train    : {X_train.shape}  ({len(X_train)/total:.1%})")
    print(f"  X_val      : {X_val.shape}  ({len(X_val)/total:.1%})")
    print(f"  X_test     : {X_test.shape}  ({len(X_test)/total:.1%})")
    print(f"\n  y_train fraud ratio : {y_train.mean():.4f}")
    print(f"  y_val   fraud ratio : {y_val.mean():.4f}")
    print(f"  y_test  fraud ratio : {y_test.mean():.4f}")
    print(f"\n  Features : {list(X_train.columns)}")
    print("----------------------------------------------------")
