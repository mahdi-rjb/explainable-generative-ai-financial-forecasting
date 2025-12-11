import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_prices(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw price file not found at {raw_path}")

    df = pd.read_csv(raw_path)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the raw price file")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Clean numeric columns
    price_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in price_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    df = df.drop_duplicates(subset=["Date"])

    return df


def add_return_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log returns and simple technical indicators.
    """
    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in the raw price file")

    df = df.copy()

    # Daily log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Simple moving averages on closing price
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # Rolling volatility of returns (for example over 20 days)
    df["rolling_vol_20"] = df["log_return"].rolling(window=20).std()

    # Drop the first rows where features are not defined
    df = df.dropna().reset_index(drop=True)

    return df


def main():
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "02_Data" / "Raw"
    processed_dir = project_root / "02_Data" / "Processed"

    raw_path = raw_dir / "sp500_prices_raw.csv"
    processed_path = processed_dir / "sp500_prices_clean.csv"

    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw prices from: {raw_path}")
    df_raw = load_raw_prices(raw_path)

    print("Adding log returns and technical features")
    df_clean = add_return_and_features(df_raw)

    print(f"Resulting shape: {df_clean.shape}")
    print("Preview of cleaned data:")
    print(df_clean.head())

    df_clean.to_csv(processed_path, index=False)
    print(f"Saved cleaned prices to: {processed_path}")


if __name__ == "__main__":
    main()
