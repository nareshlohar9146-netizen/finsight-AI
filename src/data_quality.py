import json
import argparse
from pathlib import Path

import pandas as pd


def load_dataset(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    return pd.read_csv(path)


def summarize_dataset(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_application_ids": int(
            df["SK_ID_CURR"].duplicated().sum()
        ),
        "missing_cells": int(df.isnull().sum().sum())
    }


def build_missing_report(df):
    report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": df.isnull().mean() * 100
    })

    return report.sort_values(
        "missing_percentage",
        ascending=False
    )


def build_target_report(df, target_column):
    counts = df[target_column].value_counts(dropna=False)
    percentages = (
        df[target_column]
        .value_counts(normalize=True, dropna=False)
        * 100
    )

    return {
        "counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "percentages": {
            str(key): round(float(value), 2)
            for key, value in percentages.items()
        }
    }


def save_json_report(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def main():
    print("Script started! Parsing arguments...")
    parser = argparse.ArgumentParser(
        description="FinSight AI data-quality report"
    )

    parser.add_argument(
        "--input",
        default="data/raw/application_train.csv",
        help="Path to the training CSV"
    )

    parser.add_argument(
        "--output",
        default="reports/data_quality_summary.json",
        help="Path for the JSON summary"
    )

    args = parser.parse_args()

    df = load_dataset(args.input)

    summary = summarize_dataset(df)

    missing_report = build_missing_report(df)
    missing_report.to_csv(
        "reports/missing_values.csv",
        index=True
    )

    target_report = build_target_report(df, "TARGET")

    report = {
        "dataset_summary": summary,
        "target_distribution": target_report
    }

    save_json_report(report, args.output)

    print("Dataset summary:")
    print(summary)

    print("\nTarget distribution:")
    print(target_report)

    print("\nTop 10 columns by missing percentage:")
    print(missing_report.head(10).round(2))

    print("\nData-quality report generated successfully.")


if __name__ == "__main__":
    main()