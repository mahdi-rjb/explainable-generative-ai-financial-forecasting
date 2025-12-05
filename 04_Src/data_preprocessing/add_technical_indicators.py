import os
import pandas as pd
import numpy as np
import ta


def add_technical_indicators(
    input_path: str = "../../02_Data/Processed/sp500_features_step1.csv",
    output_path: str = "../../02_Data/Processed/sp500_features_step2.csv"
):
    """
    Load cleaned price dataset and compute standard technical indicators:
      - SMA (10, 50)
      - Rolling volatility (10-day)
      - RSI (14)
      - MACD (12, 26, 9)
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Simple Moving Averages
    df["sma_10"] = df["adj_close"].rolling(window=10, min_periods=10).mean()
    df["sma_50"] = df["adj_close"].rolling(window=50, min_periods=50).mean()

    # 10-day rolling volatility based on returns
    df["volatility_10"] = df["return"].rolling(window=10, min_periods=10).std()

    # RSI (14-day)
    df["rsi_14"] = ta.momentum.rsi(df["adj_close"], window=14, fillna=False)

    # MACD (12, 26) and signal (9)
    macd = ta.trend.MACD(
        close=df["adj_close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
        fillna=False
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Drop early NaN rows caused by rolling windows
    df = df.dropna().reset_index(drop=True)

    # Save final feature file
    output_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    df.to_csv(output_abs, index=False)

    print("Technical indicators added.")
    print("Shape:", df.shape)
    print("Saved to:", output_abs)

    return df


if __name__ == "__main__":
    add_technical_indicators()
