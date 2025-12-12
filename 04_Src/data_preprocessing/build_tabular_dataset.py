import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_daily_features(path: Path) -> pd.DataFrame:
    """
    Load daily merged features for S and P five hundred.

    Expected columns include:
    Date
    log_return
    technical and sentiment features.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found at {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("Expected a Date column")
    if "log_return" not in df.columns:
        raise ValueError("Expected a log_return column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def build_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a supervised learning frame where

    X at day t predicts log_return at day t plus 1.
    """
    df = df.copy()

    # Target: next day log_return
    df["target_next_return"] = df["log_return"].shift(-1)

    # Drop last row where target is missing
    df = df.dropna(subset=["target_next_return"]).reset_index(drop=True)

    return df


def split_by_date(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    Split daily frame into train, validation and test sets based on Date.
    """
    dates = pd.to_datetime(df["Date"])

    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end

    return {
        "train": df.loc[train_mask].reset_index(drop=True),
        "val": df.loc[val_mask].reset_index(drop=True),
        "test": df.loc[test_mask].reset_index(drop=True),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "02_Data" / "Processed"

    features_path = processed_dir / "sp500_features_daily.csv"

    print(f"Loading features from: {features_path}")
    df = load_daily_features(features_path)

    df_supervised = build_supervised_frame(df)
    print("Supervised frame shape:", df_supervised.shape)

    # Date based splits consistent with sequence setup
    train_end = pd.Timestamp("2022-12-31")
    val_end = pd.Timestamp("2023-12-31")

    splits = split_by_date(df_supervised, train_end=train_end, val_end=val_end)

    for name, part in splits.items():
        out_path = processed_dir / f"sp500_tabular_{name}.csv"
        part.to_csv(out_path, index=False)
        print(f"{name} set shape: {part.shape} saved to {out_path}")


if __name__ == "__main__":
    main()
