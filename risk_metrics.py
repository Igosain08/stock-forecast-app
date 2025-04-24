import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def calculate_returns(prices):
    """Calculate daily returns from price series"""
    return prices.pct_change().dropna()

def calculate_metrics(returns):
    """Calculate common risk and performance metrics"""
    # Annualization factor for daily data
    annual_factor = 252
    
    # Convert to numpy for calculations
    returns_np = returns.values
    
    # Basic metrics
    cumulative_return = (1 + returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (annual_factor / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(annual_factor)
    
    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    sharpe_ratio = (annual_return) / annual_volatility
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    max_drawdown = drawdowns.min()
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns_np, 5)
    var_99 = np.percentile(returns_np, 1)
    
    # Results dictionary
    metrics = {
        'Cumulative Return': f"{cumulative_return:.2%}",
        'Annual Return': f"{annual_return:.2%}",
        'Annual Volatility': f"{annual_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Value at Risk (95%)': f"{var_95:.2%}",
        'Value at Risk (99%)': f"{var_99:.2%}"
    }
    
    return metrics

def plot_drawdown(returns):
    """Plot cumulative returns and drawdowns"""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    
    fig = go.Figure()
    # Cumulative returns
    fig.add_trace(go.Scatter(
        x=cum_returns.index, 
        y=cum_returns, 
        mode='lines',
        name='Cumulative Return',
        line=dict(color='green')
    ))
    
    # Add Running Max as reference
    fig.add_trace(go.Scatter(
        x=running_max.index, 
        y=running_max, 
        mode='lines',
        name='Running Max',
        line=dict(color='blue', dash='dash')
    ))
    
    fig.update_layout(
        title='Cumulative Return and Running Maximum',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Second figure for drawdowns
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=drawdowns.index, 
        y=drawdowns, 
        mode='lines',
        name='Drawdown',
        line=dict(color='red')
    ))
    
    # Add zero line
    dd_fig.add_hline(y=0, line_dash="solid", line_color="gray")
    
    dd_fig.update_layout(
        title='Drawdowns Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown',
    )
    
    return fig, dd_fig

def display_risk_analysis(stock_data):
    """Display comprehensive risk analysis"""
    # Calculate returns
    returns = calculate_returns(stock_data['Close'])
    
    # Calculate metrics
    metrics = calculate_metrics(returns)
    
    # Display metrics
    st.subheader("Risk and Performance Metrics")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Display metrics in two columns
    metrics_keys = list(metrics.keys())
    half_point = len(metrics_keys) // 2 + len(metrics_keys) % 2
    
    for i, key in enumerate(metrics_keys[:half_point]):
        col1.metric(key, metrics[key])
    
    for i, key in enumerate(metrics_keys[half_point:]):
        col2.metric(key, metrics[key])
    
    # Plot drawdowns
    st.subheader("Returns and Drawdowns Analysis")
    cum_fig, dd_fig = plot_drawdown(returns)
    st.plotly_chart(cum_fig)
    st.plotly_chart(dd_fig)
    
    # Monthly Returns Heatmap
    # Monthly Returns Heatmap
    st.subheader("Monthly Returns Heatmap")

# Convert returns to monthly and create a pivot table
# Change from 'M' to 'ME' to resolve the deprecation warning
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns.index = monthly_returns.index.strftime('%b-%Y')

    
    # Get year and month
    monthly_returns_list = []
    for date_str in monthly_returns.index:
        month, year = date_str.split('-')
        monthly_returns_list.append({
            'Month': month,
            'Year': year,
            'Return': monthly_returns[date_str]
        })
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame(monthly_returns_list)
    if not monthly_df.empty and len(monthly_df['Year'].unique()) > 1:
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        # Plot heatmap using Plotly
        z = pivot.values
        x = pivot.columns.tolist()
        y = pivot.index.tolist()
        
        # Custom colorscale (red for negative, green for positive)
        colorscale = [
            [0, 'rgb(255,0,0)'],       # Red for most negative
            [0.5, 'rgb(255,255,255)'], # White for zero
            [1, 'rgb(0,128,0)']        # Green for most positive
        ]
        
        heat_fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            text=[[f"{val:.2%}" for val in row] for row in z],
            texttemplate="%{text}",
            textfont={"size":10},
            zmin=-0.15,  # Adjust min/max for better color contrast
            zmax=0.15,
            colorbar=dict(title='Monthly Return')
        ))
        
        heat_fig.update_layout(
            title='Monthly Returns by Year',
            xaxis_title='Month',
            yaxis_title='Year',
        )
        
        st.plotly_chart(heat_fig)
    else:
        st.info("Not enough data for monthly returns heatmap. Need at least two years of data.")