# fetch_news_and_sentiment.py
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

NEWSAPI_KEY = "<YOUR_NEWSAPI_KEY>"  # get free key for dev
analyzer = SentimentIntensityAnalyzer()

def fetch_news(query="S&P 500 OR S&P500 OR stock market", from_date="2020-01-01", to_date=None, page_size=100):
    if to_date is None:
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
    all_articles = []
    url = "https://newsapi.org/v2/everything"
    page = 1
    while True:
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "pageSize": page_size,
            "page": page,
            "apiKey": NEWSAPI_KEY,
            "sortBy": "publishedAt"
        }
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print("NewsAPI error", r.status_code, r.text); break
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break
        all_articles.extend(articles)
        if len(articles) < page_size: break
        page += 1
    df = pd.DataFrame(all_articles)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    return df

def compute_vader_sentiment(df):
    df = df.copy()
    df['headline'] = df['title'].fillna('') + ". " + df['description'].fillna('')
    df['vader_scores'] = df['headline'].apply(lambda t: analyzer.polarity_scores(t))
    df = pd.concat([df.drop(columns=['vader_scores']), df['vader_scores'].apply(pd.Series)], axis=1)
    return df

if __name__ == "__main__":
    df = fetch_news(from_date="2020-01-01", to_date="2025-11-25")
    df2 = compute_vader_sentiment(df)
    df2.to_csv("../02_Data/Raw/news_raw.csv", index=False)
    print("Saved news_raw.csv")
