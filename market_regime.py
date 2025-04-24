import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from sklearn.cluster import KMeans

def detect_market_regime(stock_data, window=20):
    """
    Detect market regimes using volatility and trend
    
    Parameters:
    -----------
    stock_data: DataFrame with stock price data
    window: Window size for calculations
    
    Returns:
    --------
    DataFrame with market regimes
    """
    df = stock_data.copy()
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Calculate volatility (rolling standard deviation)
    df['volatility'] = df['returns'].rolling(window=window).std()
    
    # Calculate trend (rolling mean of returns)
    df['trend'] = df['returns'].rolling(window=window).mean()
    
    # Z-score standardization for both metrics
    df['volatility_z'] = (df['volatility'] - df['volatility'].mean()) / df['volatility'].std()
    df['trend_z'] = (df['trend'] - df['trend'].mean()) / df['trend'].std()
    
    # Define market regimes
    # High volatility, positive trend: Bull volatile
    # High volatility, negative trend: Bear volatile
    # Low volatility, positive trend: Bull quiet
    # Low volatility, negative trend: Bear quiet
    
    vol_threshold = 0.5  # Z-score threshold for high/low volatility
    trend_threshold = 0  # Z-score threshold for positive/negative trend
    
    conditions = [
        (df['volatility_z'] > vol_threshold) & (df['trend_z'] > trend_threshold),
        (df['volatility_z'] > vol_threshold) & (df['trend_z'] <= trend_threshold),
        (df['volatility_z'] <= vol_threshold) & (df['trend_z'] > trend_threshold),
        (df['volatility_z'] <= vol_threshold) & (df['trend_z'] <= trend_threshold)
    ]
    
    regimes = ['Bull Volatile', 'Bear Volatile', 'Bull Quiet', 'Bear Quiet']
    
    df['market_regime'] = np.select(conditions, regimes, default='Unknown')
    
    # Clean up NaN values
    df = df.dropna()
    
    return df

def plot_market_regimes(regime_df):
    """
    Plot market regimes
    
    Parameters:
    -----------
    regime_df: DataFrame with market regimes
    
    Returns:
    --------
    Plotly figure
    """
    # Create scatter plot of volatility vs trend
    fig = go.Figure()
    
    # Define colors for each regime
    regime_colors = {
        'Bull Volatile': 'rgba(0, 255, 0, 0.7)',    # Green with transparency
        'Bear Volatile': 'rgba(255, 0, 0, 0.7)',    # Red with transparency
        'Bull Quiet': 'rgba(144, 238, 144, 0.7)',   # Light green with transparency
        'Bear Quiet': 'rgba(255, 182, 193, 0.7)'    # Light red/pink with transparency
    }
    
    # Add traces for each regime
    for regime in regime_colors:
        mask = regime_df['market_regime'] == regime
        fig.add_trace(go.Scatter(
            x=regime_df.loc[mask, 'volatility_z'],
            y=regime_df.loc[mask, 'trend_z'],
            mode='markers',
            name=regime,
            marker=dict(
                color=regime_colors[regime],
                size=8,
                line=dict(width=1)
            )
        ))
    
    # Add axis lines
    fig.add_shape(
        type="line",
        x0=-3, y0=0, x1=3, y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=0.5, y0=-3, x1=0.5, y1=3,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Update layout
    fig.update_layout(
        title='Market Regimes: Volatility vs Trend',
        xaxis_title='Volatility (Z-score)',
        yaxis_title='Trend (Z-score)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Create a time series plot showing price and regimes
    ts_fig = go.Figure()
    
    # Add price line
    ts_fig.add_trace(go.Scatter(
        x=regime_df.index,
        y=regime_df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add colored backgrounds for regimes
    for i in range(len(regime_df) - 1):
        regime = regime_df['market_regime'].iloc[i]
        ts_fig.add_vrect(
            x0=regime_df.index[i],
            x1=regime_df.index[i+1],
            fillcolor=regime_colors.get(regime, 'gray'),
            opacity=0.3,
            layer="below",
            line_width=0
        )
    
    # Update layout
    ts_fig.update_layout(
        title='Stock Price with Market Regime Overlay',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=False
    )
    
    return fig, ts_fig

def get_optimal_model_for_regime(regime):
    """
    Get optimal model type for a given market regime
    
    Parameters:
    -----------
    regime: Market regime string
    
    Returns:
    --------
    Recommended model type
    """
    # Define optimal models for each regime
    regime_models = {
        'Bull Volatile': 'hybrid',   # Hybrid models handle volatile uptrends better
        'Bear Volatile': 'gru',      # GRU models handle volatile downtrends better
        'Bull Quiet': 'lstm',        # LSTM models work well in steady uptrends
        'Bear Quiet': 'gru'          # GRU models work well in steady downtrends
    }
    
    return regime_models.get(regime, 'lstm')  # Default to LSTM if regime unknown

def get_current_regime(regime_df):
    """
    Get the current market regime
    
    Parameters:
    -----------
    regime_df: DataFrame with market regimes
    
    Returns:
    --------
    Current market regime
    """
    if len(regime_df) > 0:
        return regime_df['market_regime'].iloc[-1]
    else:
        return 'Unknown'