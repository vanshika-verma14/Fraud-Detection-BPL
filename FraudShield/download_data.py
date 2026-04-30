"""
FraudShield — Dataset Downloader
=================================
Downloads the Credit Card Fraud Detection 2023 dataset from Kaggle
using kagglehub and places it in the data/ folder.

Usage:
    python download_data.py
"""

import os
import shutil
import kagglehub


def download_dataset():
    """Download the dataset from Kaggle and copy it to data/."""

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    dest_file = os.path.join(data_dir, "creditcard_2023.csv")

    # Skip if already downloaded
    if os.path.isfile(dest_file):
        size_mb = os.path.getsize(dest_file) / (1024 * 1024)
        print(f"Dataset already exists at: {dest_file} ({size_mb:.1f} MB)")
        print("Delete it and re-run if you want to re-download.")
        return dest_file

    print("Downloading dataset from Kaggle...")
    print("(You may need to authenticate — see instructions below)\n")

    # Download via kagglehub
    path = kagglehub.dataset_download(
        "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
    )

    print(f"\nDownloaded to: {path}")

    # Find the CSV file in the downloaded folder
    source_file = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                source_file = os.path.join(root, f)
                break
        if source_file:
            break

    if source_file is None:
        print("ERROR: No CSV file found in the downloaded dataset!")
        print(f"Check the contents of: {path}")
        return None

    # Copy to data/ folder
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy2(source_file, dest_file)

    size_mb = os.path.getsize(dest_file) / (1024 * 1024)
    print(f"\nDataset copied to: {dest_file} ({size_mb:.1f} MB)")
    print("You're all set! Now run: python -m src.train")

    return dest_file


if __name__ == "__main__":
    print("=" * 55)
    print("  FraudShield — Dataset Downloader")
    print("=" * 55)
    print()

    result = download_dataset()

    if result:
        print("\n" + "=" * 55)
        print("  SUCCESS! Next steps:")
        print("  1. python -m src.train        (train models)")
        print("  2. python -m src.evaluate     (evaluate models)")
        print("  3. python -m pytest tests/ -v (run tests)")
        print("=" * 55)
    else:
        print("\n" + "=" * 55)
        print("  FAILED — See error above.")
        print()
        print("  If authentication fails, do ONE of these:")
        print()
        print("  Option A: Set environment variable")
        print("    set KAGGLE_USERNAME=your_username")
        print("    set KAGGLE_KEY=your_api_key")
        print()
        print("  Option B: Create kaggle.json")
        print("    1. Go to kaggle.com → Account → Create API Token")
        print("    2. Save kaggle.json to C:\\Users\\YourName\\.kaggle\\")
        print("=" * 55)
