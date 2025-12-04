import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def extract_sentiment(
    input_path: str = "../../02_Data/Raw/news_raw.csv",
    output_path: str = "../../02_Data/Processed/news_sentiment_recent.csv",
):
    """
    Load raw news articles and compute sentiment scores using VADER.
    Output: cleaned dataset with sentiment columns.
    """

    analyzer = SentimentIntensityAnalyzer()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Combine title + description into a single text field
    df["headline"] = (
        df["title"].fillna("").astype(str) + ". " +
        df["description"].fillna("").astype(str)
    ).str.strip()

    # Compute sentiment scores
    scores = df["headline"].apply(analyzer.polarity_scores)

    sentiment_df = scores.apply(pd.Series)
    sentiment_df.columns = ["neg", "neu", "pos", "compound"]

    df = pd.concat([df, sentiment_df], axis=1)

    # Clean datetime field
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    # Drop rows with no datetime
    df = df.dropna(subset=["publishedAt"])

    # Output folder
    output_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    df.to_csv(output_abs, index=False)

    print(f"Sentiment extraction complete.")
    print(f"Rows: {len(df)}")
    print(f"Saved to: {output_abs}")

    return df


if __name__ == "__main__":
    extract_sentiment()
