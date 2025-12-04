"""
Robust price data cleaner for S&P500 daily CSVs.

This script is defensive:
- Accepts several common column name variants (Date, DATE, date; Adj Close, adj_close; etc.)
- Attempts to infer separators and date formats
- Provides detailed diagnostics when something is missing
- Saves a cleaned features CSV with core columns and a 1-day-ahead target

Edit the DEFAULT_INPUT_PATH below if your CSV is elsewhere.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict

# Default paths (update if your project location differs)
PROJECT_ROOT = r"C:\Users\Livkorg\PycharmProjects\explainable-generative-ai-financial-forecasting"
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "02_Data", "Raw", "sp500_daily.csv")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "02_Data", "Processed", "sp500_features_step1.csv")


# Map common column name variants to canonical names
_COLUMN_MAP = {
    "date": ["date", "Date", "DATE"],
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low": ["low", "Low", "LOW"],
    "close": ["close", "Close", "CLOSE"],
    "adj_close": ["adj_close", "Adj Close", "Adj_Close", "adjusted_close", "adjusted close", "AdjClose"],
    "volume": ["volume", "Volume", "VOLUME"]
}


def _find_cols(actual_cols):
    """
    Return a dict mapping canonical -> actual column name (or None if not present)
    """
    mapping: Dict[str, str] = {}
    lower_map = {c.lower(): c for c in actual_cols}
    for canon, variants in _COLUMN_MAP.items():
        found = None
        for v in variants:
            if v in actual_cols:
                found = v
                break
            # check lowercase matches
            if v.lower() in lower_map:
                found = lower_map[v.lower()]
                break
        mapping[canon] = found
    return mapping


def clean_price_data(input_path: str = DEFAULT_INPUT_PATH,
                     output_path: str = DEFAULT_OUTPUT_PATH):
    # Diagnostics: check file existence (absolute & relative)
    if not os.path.exists(input_path):
        # try relative path from current working dir
        alt = os.path.join(os.getcwd(), input_path)
        if os.path.exists(alt):
            input_path = alt
        else:
            raise FileNotFoundError(f"Input file not found at:\n  {input_path}\nTried alternative:\n  {alt}")

    # Try reading with common encodings and separators (comma, semicolon)
    read_attempts = []
    df = None
    for sep in [",", ";", "\t"]:
        try:
            df_try = pd.read_csv(input_path, sep=sep, engine="python")
            # minimal sanity: must have at least 3 columns if this worked
            if df_try.shape[1] >= 3:
                df = df_try
                read_attempts.append((sep, True, df.shape))
                break
            else:
                read_attempts.append((sep, False, df_try.shape))
        except Exception as e:
            read_attempts.append((sep, False, str(e)))

    if df is None:
        print("Failed to read CSV with common separators. Attempts:")
        for attempt in read_attempts:
            print(" ", attempt)
        raise RuntimeError("Unable to read input CSV. Check file format and encoding.")

    # Show initial diagnostics
    print("Initial read success.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    # Find column mapping
    colmap = _find_cols(df.columns.tolist())
    print("Detected column mapping (canonical -> actual or None):")
    for k, v in colmap.items():
        print(f"  {k:10s} -> {v}")

    # Ensure we have at least a date and an adjusted close or close
    if colmap["date"] is None:
        raise ValueError("No date column detected. Please ensure your CSV has a 'Date' column.")
    if colmap["adj_close"] is None and colmap["close"] is None:
        raise ValueError("No 'Adj Close' or 'Close' column detected. Rename column to 'Adj Close' or 'Close'.")

    # Rename to canonical names
    rename_map = {}
    for canon, actual in colmap.items():
        if actual is not None:
            rename_map[actual] = canon
    df = df.rename(columns=rename_map)

    # Parse dates (try common formats)
    try:
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')
    except Exception:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Show rows with invalid dates (if any)
    n_invalid_dates = df['date'].isna().sum()
    if n_invalid_dates > 0:
        print(f"Warning: {n_invalid_dates} rows have invalid/missing dates and will be dropped.")
        # Show sample problematic rows
        print(df[df['date'].isna()].head(5))

    # Fill adj_close from close if necessary
    if 'adj_close' not in df.columns and 'close' in df.columns:
        df['adj_close'] = df['close']

    # Keep only relevant cols (if present)
    keep_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    available = [c for c in keep_cols if c in df.columns]
    df = df[available + [c for c in df.columns if c not in available]]  # keep original extras too

    # Drop rows with missing date or adj_close
    df = df.dropna(subset=['date', 'adj_close'])
    df = df.sort_values('date').reset_index(drop=True)

    # Convert numeric columns to numeric (coerce)
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute returns
    df['return'] = df['adj_close'].pct_change()
    # replace infinite values if any
    df['return'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Log return
    df['log_return'] = np.log1p(df['return'].fillna(0))

    # Target: next day return
    df['target_return_1d'] = df['return'].shift(-1)
    df = df.dropna(subset=['target_return_1d'])  # remove last row (no future)

    # Diagnostics summary
    print("Post-cleaning shape:", df.shape)
    print("Date range:", df['date'].min(), "->", df['date'].max())
    print("Return stats (describe):")
    print(df['return'].describe())

    # Save output
    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    df.to_csv(out_abs, index=False)
    print(f"Saved cleaned file to: {out_abs}")

    # Also print head to help debug
    print("Sample rows:")
    print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    try:
        clean_price_data()
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)
