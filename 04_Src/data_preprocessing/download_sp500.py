import os
from datetime import datetime
import pandas as pd
import yfinance as yf


def download_sp500(
    start_date: str = "2020-01-01",
    end_date: str = None,
    output_path: str = "../../02_Data/Raw/sp500_daily.csv"
):
    """
    Download historical daily price data for the S&P 500 index (^GSPC) using yfinance.

    Parameters
    ----------
    start_date : str
        Start date for data download (YYYY-MM-DD).
    end_date : str
        End date for data download (YYYY-MM-DD). If None, defaults to today's date.
    output_path : str
        Path where the CSV file will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the downloaded S&P 500 data.
    """

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    ticker = "^GSPC"

    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )
    except Exception as e:
        raise RuntimeError(f"Error downloading data from yfinance: {e}")

    if df.empty:
        raise ValueError("Downloaded dataset is empty. Check internet connection or ticker symbol.")

    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)

    output_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    df.to_csv(output_abs, index=False)

    print("S&P 500 daily data downloaded successfully.")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Saved to: {output_abs}")

    return df


if __name__ == "__main__":
    download_sp500()
