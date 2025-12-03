# indicators.py
import pandas as pd
import ta

def add_technical_indicators(df):
    df = df.copy()
    df['return'] = df['Adj Close'].pct_change()
    df['log_return'] = (df['Adj Close'] / df['Adj Close'].shift(1)).apply(lambda x: np.log(1+x) if pd.notna(x) else 0)
    df['SMA_10'] = df['Adj Close'].rolling(10).mean()
    df['SMA_50'] = df['Adj Close'].rolling(50).mean()
    df['volatility_10'] = df['return'].rolling(10).std()
    # RSI using ta
    df['rsi'] = ta.momentum.rsi(df['Adj Close'], window=14, fillna=True)
    df = df.dropna()
    return df
