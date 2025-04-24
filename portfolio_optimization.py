import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco

def calculate_portfolio_performance(weights, returns):
    """
    Calculate portfolio performance metrics
    
    Parameters:
    -----------
    weights: Asset weights
    returns: Asset returns
    
    Returns:
    --------
    Dictionary with portfolio metrics
    """
    # Expected portfolio return (annualized)
    portfolio_return = np.sum(returns.mean() * weights) * 252
    
    # Expected portfolio volatility (annualized)
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe': sharpe_ratio
    }

def negative_sharpe(weights, returns):
    """
    Return negative Sharpe ratio for minimization
    """
    return -calculate_portfolio_performance(weights, returns)['sharpe']

def optimize_portfolio(stock_returns, constraints=None):
    """
    Optimize portfolio weights using Modern Portfolio Theory
    
    Parameters:
    -----------
    stock_returns: DataFrame with asset returns
    constraints: List of constraints for optimization
    
    Returns:
    --------
    Dictionary with optimization results
    """
    # Number of assets
    n_assets = len(stock_returns.columns)
    
    # Initial guess (equal weighting)
    init_weights = np.array([1.0 / n_assets] * n_assets)
    
    # Bounds for weights (0% to 100%)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Constraint: weights sum to 1
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Optimize portfolio
    opt_results = sco.minimize(
        negative_sharpe,
        init_weights,
        args=(stock_returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimal weights
    optimal_weights = opt_results['x']
    
    # Calculate performance of optimal portfolio
    optimal_performance = calculate_portfolio_performance(optimal_weights, stock_returns)
    
    return {
        'weights': optimal_weights,
        'performance': optimal_performance
    }

def generate_efficient_frontier(stock_returns, n_portfolios=1000):
    """
    Generate efficient frontier for portfolio visualization
    
    Parameters:
    -----------
    stock_returns: DataFrame with asset returns
    n_portfolios: Number of portfolios to simulate
    
    Returns:
    --------
    DataFrame with portfolio simulations
    """
    # Number of assets
    n_assets = len(stock_returns.columns)
    
    # Arrays to store results
    results = np.zeros((n_portfolios, 3 + n_assets))
    
    # Generate random portfolio weights
    for i in range(n_portfolios):
        # Random weights
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_performance(weights, stock_returns)
        
        # Store results
        results[i, 0] = portfolio_metrics['return']
        results[i, 1] = portfolio_metrics['volatility']
        results[i, 2] = portfolio_metrics['sharpe']
        
        # Store weights
        results[i, 3:] = weights
    
    # Convert to DataFrame
    columns = ['return', 'volatility', 'sharpe'] + list(stock_returns.columns)
    df = pd.DataFrame(results, columns=columns)
    
    return df

def plot_efficient_frontier(ef_df, optimal_portfolio=None, risk_free_rate=0.0):
    """
    Plot efficient frontier
    
    Parameters:
    -----------
    ef_df: DataFrame with efficient frontier data
    optimal_portfolio: Dictionary with optimal portfolio data
    risk_free_rate: Risk-free rate for capital allocation line
    
    Returns:
    --------
    Plotly figure
    """
    fig = go.Figure()
    
    # Add efficient frontier portfolios
    fig.add_trace(go.Scatter(
        x=ef_df['volatility'],
        y=ef_df['return'],
        mode='markers',
        marker=dict(
            color=ef_df['sharpe'],
            colorscale='Viridis',
            size=5,
            colorbar=dict(title='Sharpe Ratio')
        ),
        text=[f"Sharpe: {s:.2f}" for s in ef_df['sharpe']],
        name='Portfolios'
    ))
    
    # Add optimal portfolio if provided
    if optimal_portfolio:
        opt_vol = optimal_portfolio['performance']['volatility']
        opt_ret = optimal_portfolio['performance']['return']
        opt_sharpe = optimal_portfolio['performance']['sharpe']
        
        fig.add_trace(go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name=f'Optimal Portfolio (Sharpe: {opt_sharpe:.2f})'
        ))
        
        # Add Capital Allocation Line (CAL)
        max_vol = ef_df['volatility'].max() * 1.1
        cal_y = [risk_free_rate, opt_ret + (opt_ret - risk_free_rate) / opt_vol * max_vol]
        cal_x = [0, max_vol]
        
        fig.add_trace(go.Scatter(
            x=cal_x,
            y=cal_y,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Capital Allocation Line'
        ))
    
    # Update layout
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Expected Volatility (Annualized)',
        yaxis_title='Expected Return (Annualized)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig