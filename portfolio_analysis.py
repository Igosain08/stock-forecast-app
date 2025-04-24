import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.optimize as sco

def analyze_portfolio(tickers, weights=None, start_date=None, risk_free_rate=0.02):
    """
    Analyze a portfolio of stocks
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    weights : list, optional
        List of weights for each ticker. If None, equal weights are used.
    start_date : str, optional
        Start date for historical data. If None, uses 5 years from today.
    risk_free_rate : float, optional
        Annual risk-free rate for calculations, default is 2%
        
    Returns:
    --------
    dict
        Portfolio analysis results
    """
    try:
        # Validate inputs
        if not tickers:
            return {'error': 'No tickers provided'}
        
        # Set default start date if not provided (5 years ago)
        if not start_date:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1/len(tickers)] * len(tickers)
        
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Fetch historical data
        data = yf.download(tickers, start=start_date)['Adj Close']
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate mean daily returns and covariance
        mean_daily_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Calculate portfolio performance
        portfolio_return = np.sum(mean_daily_returns * weights) * 252  # Annualized
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
        # Calculate individual stock metrics
        stock_metrics = []
        for i, ticker in enumerate(tickers):
            if ticker in returns.columns:
                annual_return = mean_daily_returns[ticker] * 252
                annual_volatility = returns[ticker].std() * np.sqrt(252)
                stock_sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
                
                stock_metrics.append({
                    'ticker': ticker,
                    'weight': weights[i] * 100,  # Convert to percentage
                    'return': annual_return * 100,  # Convert to percentage
                    'volatility': annual_volatility * 100,  # Convert to percentage
                    'sharpe_ratio': stock_sharpe
                })
        
        # Calculate optimal portfolio (Maximum Sharpe Ratio)
        optimal_weights, optimal_metrics = optimize_portfolio(returns, risk_free_rate)
        
        # Calculate efficient frontier
        ef_returns, ef_volatilities = calculate_efficient_frontier(returns, 50)
        
        # Calculate historical cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        portfolio_cumulative_returns = cumulative_returns.dot(weights)
        
        # Format for charting
        portfolio_history = []
        for date, value in portfolio_cumulative_returns.items():
            portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': value
            })
        
        # Calculate drawdowns
        portfolio_drawdown = calculate_drawdown(portfolio_cumulative_returns)
        max_drawdown = portfolio_drawdown.min() * 100  # Convert to percentage
        
        # Prepare correlation matrix
        correlation_matrix = returns.corr().round(2).to_dict()
        
        # Prepare the final results
        results = {
            'tickers': tickers,
            'weights': weights.tolist(),
            'portfolio_return': portfolio_return * 100,  # Convert to percentage
            'portfolio_volatility': portfolio_std_dev * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'stock_metrics': stock_metrics,
            'correlation_matrix': correlation_matrix,
            'optimal_weights': optimal_weights,
            'optimal_return': optimal_metrics['return'] * 100,
            'optimal_volatility': optimal_metrics['volatility'] * 100,
            'optimal_sharpe': optimal_metrics['sharpe'],
            'efficient_frontier': {
                'returns': ef_returns,
                'volatilities': ef_volatilities
            },
            'portfolio_history': portfolio_history
        }
        
        return results
    
    except Exception as e:
        return {'error': str(e)}

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Calculate portfolio performance metrics
    """
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev > 0 else 0
    
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Return negative Sharpe Ratio for minimization
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(returns, risk_free_rate=0.02):
    """
    Find the optimal portfolio weights to maximize Sharpe Ratio
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds (each weight between 0 and 1)
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Optimize
    result = sco.minimize(
        negative_sharpe, 
        initial_weights, 
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimal weights
    optimal_weights = result['x']
    
    # Calculate performance with optimal weights
    optimal_return, optimal_std_dev, optimal_sharpe = portfolio_performance(
        optimal_weights, mean_returns, cov_matrix, risk_free_rate
    )
    
    # Format results
    optimal_metrics = {
        'return': optimal_return,
        'volatility': optimal_std_dev,
        'sharpe': optimal_sharpe
    }
    
    return optimal_weights.tolist(), optimal_metrics

def calculate_efficient_frontier(returns, num_portfolios=25):
    """
    Calculate the efficient frontier points
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    
    results = []
    
    # Generate random portfolios
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        results.append([portfolio_return, portfolio_std_dev])
    
    # Convert to numpy array and sort by volatility
    results = np.array(results)
    indices = results[:, 1].argsort()
    
    # Extract sorted returns and volatilities, convert to percentage
    returns = results[indices, 0] * 100
    volatilities = results[indices, 1] * 100
    
    return returns.tolist(), volatilities.tolist()

def calculate_drawdown(return_series):
    """
    Calculate drawdown for a return series
    """
    # Calculate cumulative wealth index
    wealth_index = return_series
    
    # Calculate previous peaks
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdown
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    
    return drawdown

def backtest_portfolio(tickers, weights, start_date, end_date=None, rebalance_frequency='M'):
    """
    Backtest a portfolio over a historical period
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    weights : list
        List of weights for each ticker
    start_date : str
        Start date for the backtest (YYYY-MM-DD)
    end_date : str, optional
        End date for the backtest. If None, uses today.
    rebalance_frequency : str, optional
        How often to rebalance: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
        Default is 'M' (monthly)
        
    Returns:
    --------
    dict
        Backtest results
    """
    try:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert weights to numpy array and normalize
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Fetch historical data
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Initialize portfolio value
        initial_value = 10000  # $10,000 initial investment
        portfolio_value = initial_value
        
        # Track portfolio values over time
        dates = returns.index
        portfolio_values = [portfolio_value]
        current_weights = weights.copy()
        
        # Create a rebalance schedule
        if rebalance_frequency == 'D':
            rebalance_dates = dates
        else:
            # Create date range with specified frequency
            date_range = pd.date_range(start=dates[0], end=dates[-1], freq=rebalance_frequency)
            rebalance_dates = [d for d in date_range if d in dates]
        
        # Create a mapping for rebalance dates
        rebalance_map = {date: date in rebalance_dates for date in dates}
        
        # Run the backtest
        positions = initial_value * current_weights
        portfolio_history = []
        rebalance_history = []
        
        for i, date in enumerate(dates[1:], 1):
            # Calculate new positions based on returns
            for j, ticker in enumerate(tickers):
                if ticker in returns.columns:
                    positions[j] *= (1 + returns[ticker].iloc[i])
            
            # Calculate new portfolio value
            portfolio_value = np.sum(positions)
            portfolio_values.append(portfolio_value)
            
            # Calculate current weights
            current_weights = positions / portfolio_value
            
            # Record portfolio value
            portfolio_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': portfolio_value
            })
            
            # Check if we need to rebalance
            if rebalance_map.get(date, False):
                # Record pre-rebalance weights
                pre_rebalance = {ticker: weight for ticker, weight in zip(tickers, current_weights)}
                
                # Rebalance to target weights
                positions = portfolio_value * weights
                current_weights = weights.copy()
                
                # Record rebalance event
                rebalance_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'pre_rebalance_weights': pre_rebalance,
                    'post_rebalance_weights': {ticker: weight for ticker, weight in zip(tickers, weights)}
                })
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Annualized return
        total_return = (portfolio_values[-1] / initial_value) - 1
        years = (dates[-1] - dates[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # Volatility
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annual_volatility
        
        # Max drawdown
        portfolio_series = pd.Series(portfolio_values, index=dates)
        drawdown = calculate_drawdown(portfolio_series / portfolio_series[0])
        max_drawdown = drawdown.min()
        
        # Benchmark comparison (S&P 500)
        spy_data = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
        spy_returns = spy_data.pct_change().dropna()
        spy_values = [initial_value]
        
        for ret in spy_returns:
            spy_values.append(spy_values[-1] * (1 + ret))
        
        # Match lengths
        min_length = min(len(portfolio_values), len(spy_values))
        portfolio_values = portfolio_values[:min_length]
        spy_values = spy_values[:min_length]
        
        # Calculate benchmark metrics
        spy_total_return = (spy_values[-1] / spy_values[0]) - 1
        spy_annualized_return = (1 + spy_total_return) ** (1/years) - 1
        spy_volatility = spy_returns.std() * np.sqrt(252)
        spy_sharpe = (spy_annualized_return - risk_free_rate) / spy_volatility
        
        # Prepare benchmark comparison
        benchmark_comparison = {
            'portfolio_return': annualized_return * 100,  # Convert to percentage
            'benchmark_return': spy_annualized_return * 100,  # Convert to percentage
            'portfolio_volatility': annual_volatility * 100,  # Convert to percentage
            'benchmark_volatility': spy_volatility * 100,  # Convert to percentage
            'portfolio_sharpe': sharpe_ratio,
            'benchmark_sharpe': spy_sharpe,
            'outperformance': (annualized_return - spy_annualized_return) * 100  # Convert to percentage
        }
        
        return {
            'initial_value': initial_value,
            'final_value': portfolio_values[-1],
            'total_return': total_return * 100,  # Convert to percentage
            'annualized_return': annualized_return * 100,  # Convert to percentage
            'volatility': annual_volatility * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'benchmark_comparison': benchmark_comparison,
            'portfolio_history': portfolio_history,
            'rebalance_history': rebalance_history
        }
    
    except Exception as e:
        return {'error': str(e)}

def generate_portfolio_recommendations(tickers=None, risk_profile='moderate', start_date=None):
    """
    Generate portfolio recommendations based on risk profile
    
    Parameters:
    -----------
    tickers : list, optional
        List of ticker symbols to consider. If None, uses a default set of major stocks
    risk_profile : str, optional
        Risk profile: 'conservative', 'moderate', or 'aggressive'
    start_date : str, optional
        Start date for historical data. If None, uses 5 years from today.
        
    Returns:
    --------
    dict
        Portfolio recommendations
    """
    try:
        # Default tickers representing major sectors if none provided
        if tickers is None:
            tickers = [
                # Technology
                'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABT', 'MRK',
                # Financials
                'JPM', 'BAC', 'WFC', 'V', 'MA',
                # Consumer
                'AMZN', 'PG', 'KO', 'PEP', 'WMT',
                # Energy & Industrial
                'XOM', 'CVX', 'CAT', 'BA', 'HON',
                # ETFs for diversification
                'SPY', 'QQQ', 'IWM', 'VWO', 'AGG'
            ]
        
        # Set default start date if not provided (5 years ago)
        if not start_date:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Fetch historical data
        data = yf.download(tickers, start=start_date)['Adj Close']
        
        # Filter out tickers with insufficient data
        valid_data = data.dropna(axis=1, thresh=int(len(data) * 0.9))
        valid_tickers = valid_data.columns.tolist()
        
        # Calculate daily returns
        returns = valid_data.pct_change().dropna()
        
        # Generate multiple portfolios (Monte Carlo simulation)
        num_portfolios = 5000
        results = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(valid_tickers))
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
            sharpe_ratio = portfolio_return / portfolio_std_dev  # Simplified Sharpe (no risk-free rate)
            
            results.append({
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_std_dev,
                'sharpe': sharpe_ratio
            })
        
        # Convert to DataFrame for easier filtering
        results_df = pd.DataFrame(results)
        
        # Define risk profiles
        risk_profiles = {
            'conservative': {'return_weight': 0.2, 'volatility_weight': 0.8},
            'moderate': {'return_weight': 0.5, 'volatility_weight': 0.5},
            'aggressive': {'return_weight': 0.8, 'volatility_weight': 0.2}
        }
        
        # Select profile
        profile = risk_profiles.get(risk_profile, risk_profiles['moderate'])
        
        # Filter portfolios based on risk profile
        if risk_profile == 'conservative':
            # For conservative, prioritize lower volatility
            candidate_portfolios = results_df.nsmallest(100, 'volatility')
            # Then pick the one with highest return from these
            best_portfolio = candidate_portfolios.nlargest(1, 'return').iloc[0]
        elif risk_profile == 'aggressive':
            # For aggressive, prioritize higher return
            candidate_portfolios = results_df.nlargest(100, 'return')
            # Then pick the one with lowest volatility from these
            best_portfolio = candidate_portfolios.nsmallest(1, 'volatility').iloc[0]
        else:  # moderate
            # For moderate, use the highest Sharpe ratio
            best_portfolio = results_df.nlargest(1, 'sharpe').iloc[0]
        
        # Get optimal weights from best portfolio
        optimal_weights = best_portfolio['weights']
        
        # Create portfolio recommendation
        portfolio_recommendation = {
            'risk_profile': risk_profile,
            'tickers': valid_tickers,
            'weights': optimal_weights.tolist(),
            'expected_return': best_portfolio['return'] * 100,  # Convert to percentage
            'expected_volatility': best_portfolio['volatility'] * 100,  # Convert to percentage
            'sharpe_ratio': best_portfolio['sharpe'],
            'allocation': []
        }
        
        # Add individual allocations
        for i, ticker in enumerate(valid_tickers):
            portfolio_recommendation['allocation'].append({
                'ticker': ticker,
                'weight': optimal_weights[i] * 100  # Convert to percentage
            })
        
        # Sort allocations by weight
        portfolio_recommendation['allocation'].sort(key=lambda x: x['weight'], reverse=True)
        
        return portfolio_recommendation
    
    except Exception as e:
        return {'error': str(e)}
