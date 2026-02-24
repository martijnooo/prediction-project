import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(ticker="GC=F", period="2y", interval="1d"):
    """
    Fetch financial data from Yahoo Finance.
    
    Args:
        ticker (str): Ticker symbol (default: Gold Futures 'GC=F')
        period (str): Data period to download (default: '2y')
        interval (str): Data interval (default: '1d')
    
    Returns:
        pd.DataFrame: DataFrame containing the fetched data
    """
    print(f"Fetching data for {ticker} over {period}...")
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            print(f"No data found for {ticker}.")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Ensure column names are flat (yfinance sometimes returns multi-index)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]
            
        print(f"Successfully fetched {len(data)} rows.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    # Test local execution
    df = fetch_data()
    if df is not None:
        print(df.head())
