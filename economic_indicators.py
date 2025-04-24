import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EconomicData:
    def __init__(self):
        # Define common economic indicators and their tickers/sources
        self.indicators = {
            'Treasury Yield 10Y': '^TNX',
            'Treasury Yield 2Y': '^IRX',  # Changed to IRX (13-week Treasury Bill)
            'Treasury Yield 30Y': '^TYX',
            'Crude Oil': 'CL=F',
            'Gold': 'GC=F',
            'US Dollar Index': 'DX=F',
            'VIX Volatility Index': '^VIX',
            'S&P 500': '^GSPC',
            'Nasdaq': '^IXIC',
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT'
        }
    
    def get_economic_indicators(self, start_date=None, end_date=None):
        """
        Get common economic indicators data
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in YYYY-MM-DD format. If None, uses 1 year ago.
        end_date : str, optional
            End date in YYYY-MM-DD format. If None, uses today.
            
        Returns:
        --------
        dict
            Economic indicators data
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date_obj = datetime.now()
            else:
                end_date_obj = pd.to_datetime(end_date)
            
            if not start_date:
                start_date_obj = end_date_obj - timedelta(days=365)
            else:
                start_date_obj = pd.to_datetime(start_date)
            
            logging.info(f"Fetching economic indicators from {start_date_obj} to {end_date_obj}")
            
            # Fetch data for all indicators
            results = {}
            
            for name, ticker in self.indicators.items():
                try:
                    logging.info(f"Fetching data for {name} ({ticker})...")
                    data = yf.download(ticker, start=start_date_obj, end=end_date_obj)
                    
                    if data.empty:
                        logging.warning(f"No data returned for {name} ({ticker})")
                        continue
                    
                    # Use 'Close' if 'Adj Close' is not available
                    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                    
                    if close_col not in data.columns:
                        logging.warning(f"Price column not available for {name} ({ticker})")
                        continue
                    
                    if len(data[close_col]) > 0:
                        # Calculate metrics
                        current_value = float(data[close_col].iloc[-1])
                        
                        # Safely calculate changes
                        change_1d = 0.0
                        if len(data) > 1:
                            change_1d = float((data[close_col].iloc[-1] / data[close_col].iloc[-2] - 1) * 100)
                            
                        change_1w = 0.0
                        if len(data) >= 5:
                            change_1w = float((data[close_col].iloc[-1] / data[close_col].iloc[-5] - 1) * 100)
                            
                        change_1m = 0.0
                        if len(data) >= 21:
                            change_1m = float((data[close_col].iloc[-1] / data[close_col].iloc[-21] - 1) * 100)
                        
                        ytd_data = data[data.index.year == end_date_obj.year]
                        change_ytd = 0.0
                        if not ytd_data.empty:
                            change_ytd = float((data[close_col].iloc[-1] / ytd_data[close_col].iloc[0] - 1) * 100)
                        
                        # Prepare the time series
                        series = []
                        for date, value in data[close_col].items():
                            # Make sure date is a datetime object before calling strftime
                            if isinstance(date, (datetime, pd.Timestamp)):
                                date_str = date.strftime('%Y-%m-%d')
                            else:
                                # Handle the case where date might be an integer or another type
                                date_str = str(date)
                            
                            try:
                                # Explicitly convert to float and handle nan/inf values
                                float_value = float(value)
                                if np.isnan(float_value) or np.isinf(float_value):
                                    float_value = 0.0
                            except:
                                float_value = 0.0
                                
                            series.append({
                                'date': date_str,
                                'value': float_value
                            })
                        
                        results[name] = {
                            'ticker': ticker,
                            'current_value': current_value,
                            'change_1d': change_1d,
                            'change_1w': change_1w,
                            'change_1m': change_1m,
                            'change_ytd': change_ytd,
                            'series': series
                        }
                except Exception as e:
                    logging.error(f"Error fetching {name}: {e}")
            
            # Group indicators by category
            categorized = {
                'interest_rates': {
                    'name': 'Interest Rates',
                    'data': {
                        k: v for k, v in results.items() 
                        if any(term in k for term in ['Treasury', 'Yield', 'Rate'])
                    }
                },
                'commodities': {
                    'name': 'Commodities',
                    'data': {
                        k: v for k, v in results.items() 
                        if any(term in k for term in ['Oil', 'Gold', 'Silver', 'Natural Gas'])
                    }
                },
                'currencies': {
                    'name': 'Currencies',
                    'data': {
                        k: v for k, v in results.items() 
                        if any(term in k for term in ['Dollar', 'Euro', 'Yen'])
                    }
                },
                'indices': {
                    'name': 'Market Indices',
                    'data': {
                        k: v for k, v in results.items() 
                        if any(term in k for term in ['S&P', 'Nasdaq', 'Dow', 'Russell', 'VIX'])
                    }
                }
            }
            
            # Fetch additional economic data from other sources
            additional_data = self._get_additional_economic_data()
            
            return {
                'as_of_date': end_date_obj.strftime('%Y-%m-%d'),
                'indicators': results,
                'categorized': categorized,
                'additional_data': additional_data
            }
        
        except Exception as e:
            logging.error(f"Error in get_economic_indicators: {e}")
            return {'error': str(e)}
    
    def _get_additional_economic_data(self):
        """Get additional economic data from other sources"""
        # Note: In a production app, you would use proper economic data APIs
        # This is a simplified implementation using static data for demonstration
        
        # Static economic indicators (typically would come from an API)
        return {
            'GDP Growth': {
                'value': '2.1%',
                'previous': '1.9%',
                'trend': 'up'
            },
            'Inflation Rate': {
                'value': '3.7%',
                'previous': '3.2%',
                'trend': 'up'
            },
            'Unemployment Rate': {
                'value': '3.8%',
                'previous': '3.9%',
                'trend': 'down'
            },
            'Federal Funds Rate': {
                'value': '5.25%-5.50%',
                'previous': '5.25%-5.50%',
                'trend': 'flat'
            }
        }
    
    def analyze_economic_impact(self, ticker_symbol, start_date=None, end_date=None):
        """
        Analyze the impact of economic indicators on a stock
        
        Parameters:
        -----------
        ticker_symbol : str
            The stock ticker symbol
        start_date : str, optional
            Start date in YYYY-MM-DD format. If None, uses 1 year ago.
        end_date : str, optional
            End date in YYYY-MM-DD format. If None, uses today.
            
        Returns:
        --------
        dict
            Analysis of economic indicators' impact on the stock
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date_obj = datetime.now()
            else:
                end_date_obj = pd.to_datetime(end_date)
            
            if not start_date:
                start_date_obj = end_date_obj - timedelta(days=365)
            else:
                start_date_obj = pd.to_datetime(start_date)
            
            logging.info(f"Analyzing economic impact on {ticker_symbol} from {start_date_obj} to {end_date_obj}")
            
            # Fetch stock data
            stock_data = yf.download(ticker_symbol, start=start_date_obj, end=end_date_obj)
            
            if stock_data.empty:
                logging.warning(f"No stock data available for {ticker_symbol}")
                return {'error': f"No stock data available for {ticker_symbol}"}
            
            # Ensure 'Adj Close' column exists
            if 'Adj Close' not in stock_data.columns:
                # Try 'Close' instead if 'Adj Close' is not available
                if 'Close' in stock_data.columns:
                    stock_data['Adj Close'] = stock_data['Close']
                    logging.info(f"Using 'Close' instead of 'Adj Close' for {ticker_symbol}")
                else:
                    logging.warning(f"Price data not available for {ticker_symbol}")
                    return {'error': f"Price data not available for {ticker_symbol}"}
            
            # Calculate daily returns
            stock_data['Return'] = stock_data['Adj Close'].pct_change()
            
            # Fetch economic indicators one by one to avoid issues
            correlations = {}
            
            for name, ticker in self.indicators.items():
                try:
                    logging.info(f"Processing indicator {name} ({ticker})...")
                    
                    # Download individual indicator data
                    indicator_data = yf.download(ticker, start=start_date_obj, end=end_date_obj)
                    
                    # Skip if data is empty
                    if indicator_data.empty:
                        logging.warning(f"No data available for {name} ({ticker})")
                        continue
                    
                    # Use 'Close' if 'Adj Close' is not available
                    close_col = 'Adj Close' if 'Adj Close' in indicator_data.columns else 'Close'
                    
                    if close_col not in indicator_data.columns:
                        logging.warning(f"Price column not available for {name} ({ticker})")
                        continue
                    
                    # Calculate daily returns for this indicator
                    indicator_returns = indicator_data[close_col].pct_change()
                    
                    # Create a DataFrame with both return series
                    combined_df = pd.DataFrame({
                        'stock': stock_data['Return'],
                        'indicator': indicator_returns
                    })
                    
                    # Drop NaN values (this is important to avoid dimension mismatch)
                    combined_df = combined_df.dropna()
                    
                    # Skip if not enough data points
                    if len(combined_df) < 10:  # Arbitrary minimum, adjust as needed
                        logging.warning(f"Not enough data points for correlation: {name}")
                        continue
                    
                    # Calculate correlation
                    correlation = combined_df['stock'].corr(combined_df['indicator'])
                    
                    # Only add if correlation is a valid number
                    if not np.isnan(correlation) and not np.isinf(correlation):
                        correlations[name] = float(correlation)
                        logging.info(f"Correlation between {ticker_symbol} and {name}: {correlation:.4f}")
                except Exception as e:
                    logging.error(f"Error processing indicator {name}: {e}")
                    continue
            
            # Identify top correlated indicators
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_indicators = sorted_correlations[:5] if len(sorted_correlations) >= 5 else sorted_correlations
            
            logging.info(f"Top correlated indicators for {ticker_symbol}: {[name for name, _ in top_indicators]}")
            
            # Prepare the final results
            results = {
                'ticker': ticker_symbol,
                'period': {
                    'start': start_date_obj.strftime('%Y-%m-%d'),
                    'end': end_date_obj.strftime('%Y-%m-%d')
                },
                'correlations': {k: float(v) for k, v in correlations.items()},
                'top_indicators': [{
                    'name': name,
                    'correlation': float(corr),
                    'relationship': 'positive' if corr > 0 else 'negative'
                } for name, corr in top_indicators],
                'interpretation': self._interpret_correlations(top_indicators, ticker_symbol)
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in analyze_economic_impact: {e}")
            return {'error': str(e)}
    
    def _interpret_correlations(self, top_indicators, ticker_symbol):
        """Generate interpretation of correlations"""
        if not top_indicators:
            return ["No significant correlations found."]
        
        interpretations = []
        
        for name, corr in top_indicators:
            strength = "strongly" if abs(corr) > 0.5 else "moderately" if abs(corr) > 0.3 else "weakly"
            direction = "positively" if corr > 0 else "negatively"
            
            interpretation = f"{ticker_symbol} is {strength} {direction} correlated with {name} ({corr:.2f})."
            
            # Add specific interpretations based on the indicator
            if 'Treasury' in name or 'Yield' in name:
                if corr > 0:
                    interpretation += f" This suggests {ticker_symbol} may benefit from rising interest rates."
                else:
                    interpretation += f" This suggests {ticker_symbol} may be sensitive to interest rate increases."
            
            elif 'VIX' in name:
                if corr > 0:
                    interpretation += f" This suggests {ticker_symbol} may perform better during periods of market volatility."
                else:
                    interpretation += f" This suggests {ticker_symbol} may perform better during periods of market stability."
            
            elif 'Dollar' in name:
                if corr > 0:
                    interpretation += f" This suggests {ticker_symbol} may benefit from a stronger US dollar."
                else:
                    interpretation += f" This suggests {ticker_symbol} may benefit from a weaker US dollar."
            
            interpretations.append(interpretation)
        
        return interpretations

def get_economic_indicators(start_date=None, end_date=None):
    """
    Get common economic indicators data
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in YYYY-MM-DD format. If None, uses 1 year ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. If None, uses today.
        
    Returns:
    --------
    dict
        Economic indicators data
    """
    econ_data = EconomicData()
    return econ_data.get_economic_indicators(start_date, end_date)

def analyze_economic_impact(ticker_symbol, start_date=None, end_date=None):
    """
    Analyze the impact of economic indicators on a stock
    
    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol
    start_date : str, optional
        Start date in YYYY-MM-DD format. If None, uses 1 year ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. If None, uses today.
        
    Returns:
    --------
    dict
        Analysis of economic indicators' impact on the stock
    """
    econ_data = EconomicData()
    return econ_data.analyze_economic_impact(ticker_symbol, start_date, end_date)
