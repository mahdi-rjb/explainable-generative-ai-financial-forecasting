import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


def fetch_news_stub(start_date: str, end_date: str, query: str = "stock market") -> pd.DataFrame:
    """
    Temporary stub for news fetching.

    This function simulates a news API by returning a small DataFrame
    with timestamps and headlines between start_date and end_date.

    It is useful for developing the rest of the pipeline before a real API
    is connected.
    """
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    dates = []
    headlines = []
    sources = []

    current = start
    while current <= end:
        # Use only weekdays to roughly mimic trading days
        if current.weekday() < 5:
            dates.append(current)
            headlines.append(f"Market update on {current.date().isoformat()} for query '{query}'")
            sources.append("stub_source")
        current += timedelta(days=1)

    df = pd.DataFrame(
        {
            "published_at": dates,
            "headline": headlines,
            "source": sources,
        }
    )

    return df


def fetch_news_newsapi(
    start_date: str,
    end_date: str,
    query: str = "stock market",
    language: str = "en",
    page_size: int = 100,
    max_pages: int = 5,
) -> pd.DataFrame:
    """
    Fetch news using the NewsAPI 'everything' endpoint.

    Notes
    -----
    - NewsAPI free plans usually only allow limited history.
    - If no key is available or the API call fails, you should fall back to the stub.

    Parameters
    ----------
    start_date : str
        ISO date string, for example "2020-01-01".
    end_date : str
        ISO date string, for example "2020-12-31".
    query : str
        Search phrase, for example "S&P 500" or "stock market".
    language : str
        Language code, for example "en".
    page_size : int
        Number of articles per page (maximum allowed by the API is usually 100).
    max_pages : int
        Safety limit on number of pages to fetch in one run.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: "published_at", "headline", "source".
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set in environment. Cannot use NewsAPI.")

    url = "https://newsapi.org/v2/everything"

    all_rows = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "from": start_date,
            "to": end_date,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"NewsAPI request failed with status {response.status_code}")
            break

        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            # No more results
            break

        for article in articles:
            published_at = article.get("publishedAt")
            title = article.get("title")
            source_name = (article.get("source") or {}).get("name", "unknown")

            all_rows.append(
                {
                    "published_at": published_at,
                    "headline": title,
                    "source": source_name,
                }
            )

        # Stop early if API indicates there are no more pages
        if len(articles) < page_size:
            break

    if not all_rows:
        print("No articles fetched from NewsAPI.")
        return pd.DataFrame(columns=["published_at", "headline", "source"])

    df = pd.DataFrame(all_rows)

    # Convert published_at to datetime
    df["published_at"] = pd.to_datetime(df["published_at"])

    return df


def save_news_raw(df: pd.DataFrame, raw_path: Path) -> None:
    """
    Save the raw news DataFrame to CSV.
    """
    os.makedirs(raw_path.parent, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw news data to: {raw_path}")


def fetch_news(
    start_date: str,
    end_date: str,
    query: str = "S&P 500",
    provider: Literal["stub", "newsapi"] = "stub",
) -> pd.DataFrame:
    """
    High level function used by the rest of the project.

    Parameters
    ----------
    start_date : str
        Start date in ISO format.
    end_date : str
        End date in ISO format.
    query : str
        Search query.
    provider : {"stub", "newsapi"}
        Which backend to use.

    Returns
    -------
    pd.DataFrame
        DataFrame with "published_at", "headline", "source".
    """
    if provider == "stub":
        return fetch_news_stub(start_date, end_date, query=query)
    elif provider == "newsapi":
        return fetch_news_newsapi(start_date, end_date, query=query)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def main(provider: Optional[str] = None) -> None:
    """
    Entry point so this can be run as a script.

    If provider is not given, it will try NewsAPI if NEWSAPI_KEY is valid,
    otherwise it will fall back to the stub.
    """
    # Load environment variables from .env if present
    load_dotenv()

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "02_Data" / "Raw"
    raw_path = raw_dir / "sp500_news_raw.csv"

    start_date = "2020-01-01"
    end_date = "2020-03-31"

    env_key = os.getenv("NEWSAPI_KEY", "").strip()

    if provider is None:
        # Auto select provider
        if env_key and env_key != "your_real_key_goes_here":
            chosen_provider = "newsapi"
        else:
            chosen_provider = "stub"
    else:
        chosen_provider = provider

    print(f"Using provider: {chosen_provider}")
    print(f"Fetching news from {start_date} to {end_date} for query 'S&P 500'")

    df_news: pd.DataFrame

    if chosen_provider == "newsapi":
        try:
            df_news = fetch_news(
                start_date=start_date,
                end_date=end_date,
                query="S&P 500",
                provider="newsapi",  # explicit
            )
            if df_news.empty:
                print("No articles returned by NewsAPI, falling back to stub data.")
                df_news = fetch_news_stub(start_date, end_date, query="S&P 500")
        except Exception as exc:
            print(f"Error when calling NewsAPI: {exc}")
            print("Falling back to stub data.")
            df_news = fetch_news_stub(start_date, end_date, query="S&P 500")
    else:
        # Stub path
        df_news = fetch_news_stub(start_date, end_date, query="S&P 500")

    print("Preview of fetched news:")
    print(df_news.head())

    save_news_raw(df_news, raw_path)


if __name__ == "__main__":
    main()
