import os
from pathlib import Path

import pandas as pd


def load_detailed_sentiment(path: Path) -> pd.DataFrame:
    """
    Load detailed news sentiment data.

    Expected columns include:
    - published_at
    - sentiment_compound
    and possibly sentiment_pos, sentiment_neg, sentiment_neu
    """
    if not path.exists():
        raise FileNotFoundError(f"Detailed sentiment file not found at {path}")

    df = pd.read_csv(path)

    if "published_at" not in df.columns:
        raise ValueError("Expected a 'published_at' column in the sentiment file")

    df["published_at"] = pd.to_datetime(df["published_at"])

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment to daily level.

    For each calendar date, compute:
    - mean compound sentiment
    - mean positive and negative sentiment
    - number of headlines
    """
    df = df.copy()
    df["date"] = df["published_at"].dt.date

    agg_dict = {
        "sentiment_compound": "mean",
    }

    if "sentiment_pos" in df.columns:
        agg_dict["sentiment_pos"] = "mean"
    if "sentiment_neg" in df.columns:
        agg_dict["sentiment_neg"] = "mean"

    grouped = (
        df.groupby("date")
        .agg(agg_dict)
        .rename(
            columns={
                "sentiment_compound": "sentiment_mean",
                "sentiment_pos": "sentiment_pos_mean",
                "sentiment_neg": "sentiment_neg_mean",
            }
        )
    )

    grouped["news_count"] = df.groupby("date")["headline"].count()

    grouped = grouped.reset_index()

    return grouped


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "02_Data" / "Processed"
    detailed_path = processed_dir / "sp500_news_sentiment_detailed.csv"
    daily_path = processed_dir / "sp500_news_sentiment_daily.csv"

    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading detailed sentiment from: {detailed_path}")
    df_detailed = load_detailed_sentiment(detailed_path)

    print("Aggregating sentiment to daily level")
    df_daily = aggregate_daily(df_detailed)

    print("Preview of daily sentiment:")
    print(df_daily.head())

    df_daily.to_csv(daily_path, index=False)
    print(f"Saved daily sentiment to: {daily_path}")


if __name__ == "__main__":
    main()
