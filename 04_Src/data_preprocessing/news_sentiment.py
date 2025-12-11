import os
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_raw_news(raw_path: Path) -> pd.DataFrame:
    """
    Load raw news data from CSV.

    Expected columns:
    - published_at: timestamp or date
    - headline: text of the news headline
    - source: news source name
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw news file not found at {raw_path}")

    df = pd.read_csv(raw_path)

    required_columns = {"published_at", "headline", "source"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Raw news file is missing required columns: {missing}")

    # Convert published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"])

    # Drop rows with empty headlines
    df = df.dropna(subset=["headline"]).reset_index(drop=True)

    return df


def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER sentiment scores for each headline.

    Adds the following columns:
    - sentiment_neg
    - sentiment_neu
    - sentiment_pos
    - sentiment_compound
    """
    analyzer = SentimentIntensityAnalyzer()

    df = df.copy()

    scores = df["headline"].apply(analyzer.polarity_scores)

    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])

    return df


def main() -> None:
    """
    Load raw news, compute sentiment scores and save detailed output.
    """
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "02_Data" / "Raw"
    processed_dir = project_root / "02_Data" / "Processed"

    raw_path = raw_dir / "sp500_news_raw.csv"
    detailed_path = processed_dir / "sp500_news_sentiment_detailed.csv"

    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw news from: {raw_path}")
    df_news = load_raw_news(raw_path)

    print(f"Computing sentiment scores for {len(df_news)} headlines")
    df_with_sentiment = add_sentiment_scores(df_news)

    print("Preview with sentiment:")
    print(df_with_sentiment.head())

    df_with_sentiment.to_csv(detailed_path, index=False)
    print(f"Saved detailed sentiment data to: {detailed_path}")


if __name__ == "__main__":
    main()
