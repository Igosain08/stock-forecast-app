import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Data fetching with caching
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol, start_date, end_date=None):
    """Fetch and cache stock data"""
    if end_date is None:
        end_date = datetime.now()
    return yf.Ticker(ticker_symbol).history(start=start_date, end=end_date)

@st.cache_data(ttl=3600)
def get_company_info(ticker_symbol):
    """Get and cache company info"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        return ticker.info
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return {'shortName': ticker_symbol}

# Performance monitoring
def time_execution(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time > 1:  # Only log if execution time is more than 1 second
            print(f"Function {func.__name__} took {execution_time:.2f} seconds to execute")
        return result
    return wrapper

# Loading management
def execute_with_loading(func, message, *args, **kwargs):
    """Execute a function with a loading spinner"""
    with st.spinner(message):
        return func(*args, **kwargs)

# Data storage and retrieval
def store_data(key, data):
    """Store data in session state with a timestamp"""
    st.session_state[key] = {
        'data': data,
        'timestamp': datetime.now()
    }

def get_data(key, max_age_minutes=60):
    """Get data from session state if it exists and is not too old"""
    if key in st.session_state:
        stored = st.session_state[key]
        age = (datetime.now() - stored['timestamp']).total_seconds() / 60
        if age < max_age_minutes:
            return stored['data']
    return None

# Format large numbers
def format_number(num):
    """Format large numbers for display"""
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num:.2f}"

# Truncate data for performance mode
def truncate_for_performance(df, performance_mode):
    """Reduce dataframe size in performance mode"""
    if not performance_mode:
        return df
    
    # If we have more than 1000 rows, sample down
    if len(df) > 1000:
        # Keep the most recent 500 rows at full resolution
        recent = df.iloc[-500:]
        
        # Sample the older data
        older = df.iloc[:-500]
        sample_rate = max(1, len(older) // 500)
        sampled = older.iloc[::sample_rate]
        
        # Combine and return
        return pd.concat([sampled, recent])
    
    return df