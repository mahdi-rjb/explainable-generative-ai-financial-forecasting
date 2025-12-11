import os
from pathlib import Path

import pandas as pd


def load_prices(path: Path) -> pd.DataFrame:
    """
    Load cleaned S and P five hundred prices.

    Expected columns include:
    - Date
    - Close
    - log_return
    plus technical features.
    """
    if not path.exists():
        raise FileNotFoundError(f"Clean price file not found at {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the prices file")

    df["Date"] = pd.to_datetime(df["Date"])
    df["date"] = df["Date"].dt.date

    return df


def load_daily_sentiment(path: Path) -> pd.DataFrame:
    """
    Load daily sentiment features.

    Expected columns:
    - date
    - sentiment_mean
    - news_count
    and possibly sentiment_pos_mean, sentiment_neg_mean.
    """
    if not path.exists():
        raise FileNotFoundError(f"Daily sentiment file not found at {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the daily sentiment file")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def merge_prices_and_sentiment(
    df_prices: pd.DataFrame,
    df_sent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily price features with daily sentiment features on the date column.

    For days without news, sentiment values are set to zero.
    """
    merged = df_prices.merge(df_sent, on="date", how="left")

    sentiment_cols = [col for col in merged.columns if col.startswith("sentiment_")]
    if "news_count" in merged.columns:
        sentiment_cols.append("news_count")

    for col in sentiment_cols:
        merged[col] = merged[col].fillna(0.0)

    merged = merged.drop(columns=["date"])

    return merged


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "02_Data" / "Processed"

    prices_path = processed_dir / "sp500_prices_clean.csv"
    daily_sent_path = processed_dir / "sp500_news_sentiment_daily.csv"
    features_path = processed_dir / "sp500_features_daily.csv"

    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading prices from: {prices_path}")
    df_prices = load_prices(prices_path)

    print(f"Loading daily sentiment from: {daily_sent_path}")
    df_sent = load_daily_sentiment(daily_sent_path)

    print("Merging prices and sentiment")
    df_features = merge_prices_and_sentiment(df_prices, df_sent)

    print("Preview of merged features:")
    print(df_features.head())

    df_features.to_csv(features_path, index=False)
    print(f"Saved merged features to: {features_path}")


if __name__ == "__main__":
    main()
