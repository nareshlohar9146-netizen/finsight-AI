
from pathlib import Path

import numpy as np
import pandas as pd

def load_dataset(file_path):
    """Load a CSV dataset and validate its basic structure."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if "TARGET" not in df.columns:
        raise ValueError("Required column TARGET is missing.")

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Required column SK_ID_CURR is missing.")

    if df["TARGET"].isnull().any():
        raise ValueError("TARGET contains missing values.")

    if not df["TARGET"].isin([0, 1]).all():
        raise ValueError("TARGET must contain only 0 and 1.")

    return df

def inspect_data_quality(df):
    """Return a summary of important data-quality checks."""

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_application_ids": int(
            df["SK_ID_CURR"].duplicated().sum()
        ),
        "missing_target": int(df["TARGET"].isna().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "infinite_values": int(
            np.isinf(
                df.select_dtypes(include="number").to_numpy()
            ).sum()
        ),
    }

    return summary

def add_financial_features(df):
    """Create financial and demographic features."""

    result = df.copy()

    required_columns = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    valid_income = result["AMT_INCOME_TOTAL"].where(
        result["AMT_INCOME_TOTAL"] > 0
    )

    result["CREDIT_INCOME_RATIO"] = (
        result["AMT_CREDIT"] / valid_income
    )

    result["ANNUITY_INCOME_RATIO"] = (
        result["AMT_ANNUITY"] / valid_income
    )

    result["AGE_YEARS"] = (
        -result["DAYS_BIRTH"] / 365.25
    )

    return result