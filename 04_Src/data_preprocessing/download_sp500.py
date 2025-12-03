# download_sp500.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def download_sp500(start="2020-01-01", end=None, out_path="../02_Data/Raw/sp500_daily.csv"):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    ticker = "^GSPC"  # S&P500 index symbol
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}, rows={len(df)}")
    return df

if __name__ == "__main__":
    download_sp500()
