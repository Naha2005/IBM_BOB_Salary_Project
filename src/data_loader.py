"""
data_loader.py
--------------
Loads and validates the salary dataset from a CSV file.
"""

import os
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'salary_data.csv')


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the salary CSV file and return a validated DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated salary data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    # ── Validate expected columns ──────────────────────────────────────────
    expected_cols = {'YearsExperience', 'Salary'}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Check for missing values ───────────────────────────────────────────
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"[WARNING] Null values detected:\n{null_counts[null_counts > 0]}")
        df = df.dropna()
        print(f"[INFO] Rows after dropping nulls: {len(df)}")

    # ── Basic type coercion ────────────────────────────────────────────────
    df['YearsExperience'] = pd.to_numeric(df['YearsExperience'], errors='coerce')
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    df = df.dropna()

    print(f"[INFO] Dataset loaded successfully: {len(df)} rows")
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a quick statistical summary of the dataset."""
    print("\n== Dataset Summary ==================================")
    print(df.describe().round(2).to_string())
    print("=====================================================\n")
