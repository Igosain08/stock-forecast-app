import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_benchmark_data(ticker_symbol, start_date, benchmark_tickers=None):
    """
    Get comparative performance data between a stock and market benchmarks
    
    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol
    start_date : str
        The start date for the comparison (YYYY-MM-DD)
    benchmark_tickers : list, optional
        List of benchmark tickers to compare against. If None, defaults to SPY, QQQ, and sector ETF
        
    Returns:
    --------
    dict
        Comparative performance data
    """
    try:
        # Get stock info to determine sector
        ticker_info = yf.Ticker(ticker_symbol).info
        sector = ticker_info.get('sector', 'Technology')
        
        # Map sector to sector ETF
        sector_etfs = {
            'Technology': 'XLK',
            'Financial Services': 'XLF',
            'Healthcare': 'XLV',
            'Communication Services': 'XLC',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Basic Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU'
        }
        
        sector_etf = sector_etfs.get(sector, 'SPY')  # Default to SPY if sector not found
        
        # Set default benchmarks if none provided
        if benchmark_tickers is None:
            benchmark_tickers = ['SPY', 'QQQ', sector_etf]
        
        # Make sure we don't duplicate the sector ETF
        if sector_etf in benchmark_tickers and sector_etf != 'SPY':
            benchmarks = benchmark_tickers
        else:
            benchmarks = benchmark_tickers + [sector_etf]
        
        # Add the main ticker to the list
        tickers_to_fetch = [ticker_symbol] + benchmarks
        
        # Fetch historical data
        data = yf.download(tickers_to_fetch, start=start_date, end=datetime.now())['Adj Close']
        
        # Calculate normalized returns (start at 100)
        normalized = data.copy()
        for ticker in normalized.columns:
            normalized[ticker] = normalized[ticker] / normalized[ticker].iloc[0] * 100
        
        # Calculate performance metrics
        performance = {}
        
        # Add stock and benchmark data
        performance['ticker'] = ticker_symbol
        performance['sector'] = sector
        performance['benchmarks'] = benchmarks
        
        # Calculate returns for different time periods
        periods = {
            '1M': 21,
            '3M': 63,
            '6M': 126,
            'YTD': 0,  # Special case, handled below
            '1Y': 252,
            '3Y': 756,
            '5Y': 1260,
            'Max': len(data)
        }
        
        returns = {}
        for period, days in periods.items():
            returns[period] = {}
            
            if period == 'YTD':
                # For YTD, get the first trading day of the year
                start_of_year = datetime(datetime.now().year, 1, 1)
                ytd_data = yf.download(tickers_to_fetch, start=start_of_year, end=datetime.now())['Adj Close']
                
                for ticker in tickers_to_fetch:
                    if ticker in ytd_data.columns:
                        first_value = ytd_data[ticker].iloc[0]
                        last_value = ytd_data[ticker].iloc[-1]
                        returns[period][ticker] = ((last_value / first_value) - 1) * 100
            else:
                # For other periods, use the last X days
                days = min(days, len(data) - 1)  # Make sure we don't exceed data length
                
                for ticker in tickers_to_fetch:
                    if ticker in data.columns:
                        first_value = data[ticker].iloc[-days] if days < len(data) else data[ticker].iloc[0]
                        last_value = data[ticker].iloc[-1]
                        returns[period][ticker] = ((last_value / first_value) - 1) * 100
        
        performance['returns'] = returns
        
        # Calculate correlations
        correlations = {}
        for benchmark in benchmarks:
            if benchmark in data.columns and ticker_symbol in data.columns:
                corr = data[ticker_symbol].corr(data[benchmark])
                correlations[benchmark] = corr
        
        performance['correlation'] = correlations
        
        # Calculate beta (against SPY or first benchmark)
        market_benchmark = 'SPY' if 'SPY' in data.columns else benchmarks[0]
        if market_benchmark in data.columns and ticker_symbol in data.columns:
            # Calculate daily returns
            stock_returns = data[ticker_symbol].pct_change().dropna()
            market_returns = data[market_benchmark].pct_change().dropna()
            
            # Make sure the lengths match
            min_length = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns[-min_length:]
            market_returns = market_returns[-min_length:]
            
            # Calculate beta
            covariance = stock_returns.cov(market_returns)
            variance = market_returns.var()
            beta = covariance / variance if variance != 0 else 1
            
            performance['beta'] = beta
        else:
            performance['beta'] = 1
        
        # Prepare time series data for charts
        performance['normalized_prices'] = normalized.reset_index().to_dict('records')
        
        return performance
    
    except Exception as e:
        return {'error': str(e)}

def get_sector_performance():
    """
    Get performance data for all sectors
    """
    try:
        # List of sector ETFs
        sector_etfs = {
            'Technology': 'XLK',
            'Financial': 'XLF',
            'Healthcare': 'XLV',
            'Communication': 'XLC',
            'Consumer Cyclical': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU'
        }
        
        # Fetch data for all sectors for the past year
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = yf.download(list(sector_etfs.values()), start=start_date)['Adj Close']
        
        # Calculate metrics for each sector
        sectors = []
        for sector_name, etf in sector_etfs.items():
            if etf in data.columns:
                # Calculate returns for different time periods
                weekly_return = ((data[etf].iloc[-1] / data[etf].iloc[-5]) - 1) * 100
                monthly_return = ((data[etf].iloc[-1] / data[etf].iloc[-21]) - 1) * 100
                quarterly_return = ((data[etf].iloc[-1] / data[etf].iloc[-63]) - 1) * 100
                ytd_start = data[etf][data[etf].index.year == datetime.now().year].iloc[0]
                ytd_return = ((data[etf].iloc[-1] / ytd_start) - 1) * 100
                yearly_return = ((data[etf].iloc[-1] / data[etf].iloc[0]) - 1) * 100
                
                sectors.append({
                    'name': sector_name,
                    'etf': etf,
                    'weekly': weekly_return,
                    'monthly': monthly_return,
                    'quarterly': quarterly_return,
                    'ytd': ytd_return,
                    'yearly': yearly_return,
                    'current_price': data[etf].iloc[-1]
                })
        
        # Sort by yearly performance
        sectors.sort(key=lambda x: x['yearly'], reverse=True)
        
        return sectors
    
    except Exception as e:
        return {'error': str(e)}

def get_market_movers():
    """
    Get top gainers, losers, and most active stocks in the market
    """
    try:
        # For a production app, you might want to use a proper market data API
        # This is a simplified version using common indices and large caps
        
        # Get data for S&P 500 components (using SPY as proxy for top holdings)
        spy_holdings = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ', 'UNH', 
                       'V', 'PG', 'HD', 'MA', 'BAC', 'XOM', 'TSLA', 'CSCO', 'PFE', 'DIS']
        
        # Get today's data and yesterday's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Get a few days in case of weekends/holidays
        
        data = yf.download(spy_holdings, start=start_date)
        
        # Calculate daily returns
        returns = data['Adj Close'].pct_change().iloc[-1] * 100
        volumes = data['Volume'].iloc[-1]
        
        # Filter out any with NaN values
        valid_returns = returns.dropna()
        valid_volumes = volumes.dropna()
        
        # Get top gainers
        gainers = valid_returns.nlargest(5)
        gainers_data = []
        for ticker, ret in gainers.items():
            price = data['Adj Close'][ticker].iloc[-1]
            gainers_data.append({
                'ticker': ticker,
                'price': price,
                'change_percent': ret,
                'volume': int(volumes[ticker]) if ticker in volumes else 0
            })
        
        # Get top losers
        losers = valid_returns.nsmallest(5)
        losers_data = []
        for ticker, ret in losers.items():
            price = data['Adj Close'][ticker].iloc[-1]
            losers_data.append({
                'ticker': ticker,
                'price': price,
                'change_percent': ret,
                'volume': int(volumes[ticker]) if ticker in volumes else 0
            })
        
        # Get most active by volume
        most_active = valid_volumes.nlargest(5)
        active_data = []
        for ticker, vol in most_active.items():
            price = data['Adj Close'][ticker].iloc[-1]
            ret = returns[ticker] if ticker in returns else 0
            active_data.append({
                'ticker': ticker,
                'price': price,
                'change_percent': ret,
                'volume': int(vol)
            })
        
        return {
            'gainers': gainers_data,
            'losers': losers_data,
            'most_active': active_data
        }
    
    except Exception as e:
        return {'error': str(e)}
