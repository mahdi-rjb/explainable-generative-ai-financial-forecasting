import os
import sys
from datetime import datetime, timedelta
import time
import requests
import pandas as pd


NEWSAPI_KEY = "29476ce93a1a46e389d2fb889613c19d"


def fetch_news(
    query: str = "stock market OR S&P 500",
    from_date: str = "2020-01-01",
    to_date: str = None,
    page_size: int = 100,
    max_pages: int = 50,
    output_path: str = "../../02_Data/Raw/news_raw.csv",
):
    """
    Fetch news articles using the NewsAPI Everything endpoint.
    Automatically adjusts from_date for free tier limits.
    """

    if NEWSAPI_KEY.startswith("<PUT"):
        raise ValueError("Please set your NewsAPI key in NEWSAPI_KEY.")

    if to_date is None:
        to_date = datetime.utcnow().strftime("%Y-%m-%d")

    # NewsAPI free tier only allows about 30 days history
    max_history_days = 28
    earliest_allowed = (datetime.utcnow() - timedelta(days=max_history_days)).strftime("%Y-%m-%d")

    if from_date < earliest_allowed:
        print(f"Adjusting from_date from {from_date} to {earliest_allowed} to satisfy free tier limits.")
        from_date = earliest_allowed

    url = "https://newsapi.org/v2/everything"

    all_articles = []
    total_requests = 0

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": NEWSAPI_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            break

        all_articles.extend(articles)
        total_requests += 1

        time.sleep(1)

        if len(articles) < page_size:
            break

    if not all_articles:
        print("No articles retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df = df.loc[:, ["source", "author", "title", "description", "url", "publishedAt", "content"]]
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    output_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    df.to_csv(output_abs, index=False)

    print(f"Retrieved {len(df)} articles across {total_requests} requests.")
    print(f"Saved to: {output_abs}")

    return df


if __name__ == "__main__":
    fetch_news()
