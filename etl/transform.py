import pandas as pd
import numpy as np

def transform_data(df):
    """
    Clean and transform the raw data.
    
    Args:
        df (pd.DataFrame): Raw dataframe from extract step
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df is None or df.empty:
        print("No data to transform.")
        return None
        
    print("Transforming data...")
    df_clean = df.copy()
    
    # Standardize column names
    # yfinance usually returns: Date, Open, High, Low, Close, Adj Close, Volume
    # We'll normalize to lowercase snake_case and remove ticker suffix if present
    new_cols = []
    for c in df_clean.columns:
        c_str = str(c).lower().replace(' ', '_')
        if '_' in c_str and 'date' not in c_str:
            # e.g. close_gc=f -> close
            c_str = c_str.split('_')[0]
        new_cols.append(c_str)
    df_clean.columns = new_cols
    
    # Ensure date column is datetime
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Handle missing values (forward fill for financial time series is common)
    df_clean.ffill(inplace=True)
    
    # Calculate simple features (e.g., daily return)
    if 'close' in df_clean.columns:
        df_clean['daily_return'] = df_clean['close'].pct_change()
        
    # Drop NAs created by feature engineering
    df_clean.dropna(inplace=True)
    
    print(f"Data transformed. Shape: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    # Test with dummy data
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'Close': [1800, 1810, 1805, 1820, 1815]
    }
    df = pd.DataFrame(data)
    print(transform_data(df))
