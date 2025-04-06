import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from monte_carlo_module import run_monte_carlo
from lstm_module import run_lstm_forecast

# --- Page setup ---
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2014-10-25"))
sample_size = st.sidebar.slider("Days to Simulate", 50, 300, 100)
iterations = st.sidebar.slider("Simulations", 100, 2000, 500)
ci_level = st.sidebar.slider("Confidence Interval (%)", 90, 99, 95)

# --- Main App ---
st.title("📊 Stock Price Forecasting App")

try:
    # --- Fetch stock data ---
    
    start_date = '2014-10-25'
    ticker = yf.Ticker(ticker_symbol)
    dataset = ticker.history(start=start_date)
    Close = dataset[['Close']]
    Close1=Close.reset_index()
    
    company_name = yf.Ticker(ticker_symbol).info.get('longName', ticker_symbol)

    # --- Show historical prices ---
    st.subheader(f"Historical Closing Prices for {company_name}")
    st.line_chart(Close)

    # --- LSTM Forecast Section ---
    st.subheader("📈 LSTM Forecast (Next Day)")
    next_prediction, mse, r2, mae = run_lstm_forecast(Close)
    st.write(f"Predicted next close price: **${next_prediction:.2f}**")
    st.write(f"Mean Squared Error (MSE): **{mse:.2f}**")
    st.write(f"R² Score: **{r2:.4f}**")
    st.write(f"Mean Absolute Error (MAE): **{mae:.2f}**")

    # --- Monte Carlo Simulation Section ---
    st.subheader("🎲 Monte Carlo Simulation")
    fig, (ci_lower, ci_upper),end_prices = run_monte_carlo(Close, sample_size, iterations, ci_level)
    st.pyplot(fig)
    st.write(f"{ci_level}% Confidence Interval for the stock price after {sample_size} days: "
             f"(${ci_lower:.2f}, ${ci_upper:.2f})")
    expected_price = np.mean(end_prices)
    st.write(f"📈 Expected price after {sample_size} days: **${expected_price:.2f}**")
    volatility = np.std(end_prices)
    st.write(f"📊 Volatility (Std Dev of simulated prices): **${volatility:.2f}**")


except Exception as e:
    st.error(f"An error occurred: {e}")