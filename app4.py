import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Import your original modules
from lstm_module import run_lstm_forecast
from monte_carlo_module import run_monte_carlo

# Import new enhancement modules
from technical_indicators import enrich_dataframe
from risk_metrics import display_risk_analysis
from model_utils import create_lstm_model, create_gru_model, create_hybrid_model

# Import NEW modules for ensemble and sentiment
from ensemble_models import create_ensemble_model, predict_next_with_ensemble
from sentiment_analysis import get_sentiment_feature

# Page setup
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# --- Page setup ---
st.title("📊 MarketScope")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.sidebar.date_input("Start Date:", value=pd.to_datetime("2014-10-25"))
sample_size = st.sidebar.slider("Days to Simulate:", 50, 300, 100)
iterations = st.sidebar.slider("Simulations:", 100, 2000, 500)
ci_level = st.sidebar.slider("Confidence Interval (%)", 80, 99, 95)

# Add model selection in sidebar
model_type = st.sidebar.selectbox(
    "Forecast Model Type",
    ["LSTM", "GRU", "Hybrid LSTM-GRU"],
    index=0,
    help="Select the type of neural network model for forecasting"
)

# Map selection to model_type parameter
model_map = {
    "LSTM": "lstm",
    "GRU": "gru",
    "Hybrid LSTM-GRU": "hybrid"
}

try:
    # --- Fetch stock data ---
    ticker = yf.Ticker(ticker_symbol)
    dataset = ticker.history(start=start_date, end=pd.Timestamp.today())
    
    # Check if we have data
    if dataset.empty:
        st.error(f"No data found for ticker {ticker_symbol}. Please check the ticker symbol.")
    else:
        # Add technical indicators
        df_with_indicators = enrich_dataframe(dataset)
        
        # Company name for display
        company_name = yf.Ticker(ticker_symbol).info.get('longName', ticker_symbol)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Price Forecast", 
            "Technical Indicators", 
            "Risk Analysis", 
            "Monte Carlo Simulation", 
            "Ensemble Models", 
            "Sentiment Analysis"
        ])
        
        with tab1:
            # --- Show historical prices ---
            st.subheader(f"Historical Closing Prices for {company_name}")
            st.line_chart(dataset['Close'])
            
            # --- LSTM Forecast Section ---
            st.subheader(f"🧠 {model_type} Forecast (Next Day)")
            
            # Run the selected model
            next_prediction, mse, r2, mae = run_lstm_forecast(dataset['Close'], model_type=model_map[model_type])
            
            # Display predictions and metrics
            st.write(f"📈Predicted next close price: **${next_prediction:.2f}**")
            st.write(f"📊Mean Squared Error (MSE): **{mse:.2f}**")
            st.write(f"📈R² Score: **{r2:.2f}**")
            st.write(f"📉Mean Absolute Error (MAE): **{mae:.2f}**")
            lower_bound = next_prediction - mae
            upper_bound = next_prediction + mae
    
            st.write(f"📊 Prediction Range (based on MAE): **${lower_bound:.2f} to ${upper_bound:.2f}**")
    
            # Initialize prediction history tracking if it doesn't exist
            if 'historical_predictions' not in st.session_state:
                st.session_state.historical_predictions = pd.DataFrame(
                    columns=['Date', 'Predicted', 'Actual']
                )
    
            # Get today's date for storing the prediction
            today = pd.Timestamp.today().strftime('%Y-%m-%d')
    
            # Check if we already stored a prediction for today
            if today not in st.session_state.historical_predictions['Date'].values:
                # Store today's prediction
                new_prediction = pd.DataFrame({
                    'Date': [today],
                    'Predicted': [next_prediction],
                    'Actual': [None]  # Will be filled in when actual data becomes available
                })  
                st.session_state.historical_predictions = pd.concat([
                    st.session_state.historical_predictions, 
                    new_prediction
                ]).reset_index(drop=True)
    
            # Update previous predictions with actual data if available
            yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
            # If yesterday is in our prediction history and we now have actual data for it
            if yesterday in st.session_state.historical_predictions['Date'].values:
                if yesterday in dataset.index.strftime('%Y-%m-%d').values:
                    # Get yesterday's actual closing price
                    yesterday_close = dataset.loc[dataset.index.strftime('%Y-%m-%d') == yesterday, 'Close'].values[0]
            
                    # Update the actual value in our tracking dataframe
                    yesterday_idx = st.session_state.historical_predictions[
                        st.session_state.historical_predictions['Date'] == yesterday
                    ].index
            
                    if len(yesterday_idx) > 0:
                        st.session_state.historical_predictions.loc[yesterday_idx[0], 'Actual'] = yesterday_close
    
            # Show prediction history if we have enough data points
            if len(st.session_state.historical_predictions.dropna()) > 2:  # Need at least a few points with actual values
                st.subheader("Recent Prediction History")
        
                # Filter to only show rows with actual values
                history_to_show = st.session_state.historical_predictions.dropna().tail(10)
        
                # Create a line chart of predicted vs actual values
                fig = px.line(history_to_show, 
                    x='Date', 
                    y=['Predicted', 'Actual'],
                    title='Previous Predictions vs Actual Prices',
                    labels={'value': 'Price ($)', 'variable': 'Series'})
        
                # Add customization to the chart
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    legend_title='',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
        
                st.plotly_chart(fig)
        
                # Calculate accuracy metrics for displayed predictions
                mean_error = (history_to_show['Actual'] - history_to_show['Predicted']).mean()
                mean_abs_error = (history_to_show['Actual'] - history_to_show['Predicted']).abs().mean()
        
                # Display accuracy metrics
                col1, col2 = st.columns(2)
                col1.metric("Avg. Prediction Error", f"${mean_error:.2f}")
                col2.metric("Avg. Absolute Error", f"${mean_abs_error:.2f}")
        
        with tab2:
            st.subheader("Technical Indicators")
            
            # Create a selection for which indicators to display
            indicator_options = ["Moving Averages", "Momentum Indicators", "Volatility Indicators"]
            selected_indicators = st.multiselect("Select indicators to display:", indicator_options, default=["Moving Averages"])
            
            # Display the selected indicators
            if "Moving Averages" in selected_indicators:
                st.subheader("Moving Averages")
                ma_fig = px.line(df_with_indicators, x=df_with_indicators.index, 
                              y=['Close', 'SMA20', 'SMA50', 'SMA200'], 
                              title=f"Moving Averages for {ticker_symbol}")
                st.plotly_chart(ma_fig)
            
            if "Momentum Indicators" in selected_indicators:
                st.subheader("RSI (Relative Strength Index)")
                rsi_fig = px.line(df_with_indicators, x=df_with_indicators.index, y='RSI', 
                               title=f"RSI for {ticker_symbol}")
                # Add reference lines at 30 and 70
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(rsi_fig)
                
                st.subheader("MACD (Moving Average Convergence Divergence)")
                macd_fig = px.line(df_with_indicators, x=df_with_indicators.index, 
                                y=['MACD', 'MACD_Signal'], 
                                title=f"MACD for {ticker_symbol}")
                st.plotly_chart(macd_fig)
            
            if "Volatility Indicators" in selected_indicators:
                st.subheader("Bollinger Bands")
                bb_fig = px.line(df_with_indicators, x=df_with_indicators.index, 
                              y=['Close', 'BB_upper', 'BB_middle', 'BB_lower'], 
                              title=f"Bollinger Bands for {ticker_symbol}")
                st.plotly_chart(bb_fig)
        
        with tab3:
            # Risk Analysis Tab
            display_risk_analysis(dataset)
        
        with tab4:
            # --- Monte Carlo Simulation Section ---
            st.subheader(f"🔮 Monte Carlo Simulation")
            
            # Run Monte Carlo simulation
            fig, (ci_lower, ci_upper), end_prices = run_monte_carlo(dataset, sample_size=sample_size, iterations=iterations, ci_level=ci_level)
            
            # Predict expected price
            expected_price = np.mean(end_prices)
            
            # Calculate volatility
            volatility = np.std(end_prices)
            
            # Display results
            st.pyplot(fig)
            st.write(f"📈 {ci_level}% Confidence Interval for the stock price after {sample_size} days: ")
            st.write(f"**${ci_lower:.2f} - ${ci_upper:.2f}**")
            
            st.write(f"📊 Expected price after {sample_size} days: **${expected_price:.2f}**")
            st.write(f"📊 Volatility (Std Dev of simulated prices): **${volatility:.2f}**")
        
        with tab5:
            # --- Ensemble Models Section ---
            st.subheader(f"🤖 Ensemble Models Forecast")
            
            # Add a button to run the ensemble forecast
            if st.button("Run Ensemble Forecast"):
                with st.spinner("Training ensemble models... This may take a moment."):
                    try:
                        # Prepare data for the ensemble model
                        from sklearn.preprocessing import MinMaxScaler
                        
                        # Extract close prices
                        close_prices = dataset['Close'].values.reshape(-1, 1)
                        
                        # Scale the data
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(close_prices)
                        
                        # Create training sequences
                        window_size = 20  # Can be adjusted via a slider if desired
                        X = []
                        y = []
                        for i in range(window_size, len(scaled_data)):
                            X.append(scaled_data[i-window_size:i, 0])
                            y.append(scaled_data[i, 0])
                        
                        # Convert to numpy arrays
                        X = np.array(X)
                        y = np.array(y)
                        
                        # Reshape input data for LSTM [samples, time steps, features]
                        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                        
                        # Split into training and testing sets (80/20)
                        train_size = int(len(X) * 0.8)
                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]
                        
                        # Create and train ensemble model
                        input_shape = (X.shape[1], X.shape[2])
                        ensemble = create_ensemble_model(X_train, y_train, X_test, y_test, input_shape)
                        
                        # Get the most recent window for prediction
                        X_recent = scaled_data[-window_size:].reshape(1, window_size, 1)
                        
                        # Make a prediction
                        next_pred = predict_next_with_ensemble(ensemble, X_recent)
                        
                        # Scale the prediction back
                        next_pred_scaled = np.array([[0] * (X.shape[2] - 1) + [next_pred]])
                        next_pred_original = scaler.inverse_transform(next_pred_scaled)[0, -1]
                        
                        # Display results
                        st.success("Ensemble Forecast Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Ensemble Prediction (Next Day)", f"${next_pred_original:.2f}")
                        
                        with col2:
                            # Compare with LSTM prediction
                            st.metric("Standard LSTM Prediction", f"${next_prediction:.2f}", 
                                    f"{((next_pred_original - next_prediction) / next_prediction) * 100:.2f}%")
                        
                        with col3:
                            st.metric("Ensemble MSE", f"{ensemble['ensemble_mse']:.6f}")
                        
                        # Display information about individual models
                        st.subheader("Model Performance Comparison")
                        
                        # Create a bar chart comparing MSEs of individual models
                        individual_mses = []
                        for i, pred in enumerate(ensemble['individual_predictions']):
                            mse_i = ((pred - y_test.reshape(-1, 1)) ** 2).mean()
                            individual_mses.append(mse_i)
                        
                        model_names = ['LSTM', 'GRU', 'Hybrid'][:len(individual_mses)]
                        model_names.append('Ensemble')
                        all_mses = individual_mses + [ensemble['ensemble_mse']]
                        
                        mse_df = pd.DataFrame({
                            'Model': model_names,
                            'MSE': all_mses
                        })
                        
                        mse_fig = px.bar(mse_df, x='Model', y='MSE', 
                                       title='Model Performance Comparison (Lower MSE is better)',
                                       color='Model')
                        st.plotly_chart(mse_fig)
                        
                    except Exception as e:
                        st.error(f"Error in ensemble forecast: {str(e)}")
            
            # Add explanation of ensemble models
            with st.expander("About Ensemble Models"):
                st.write("""
                Ensemble models combine predictions from multiple different models to produce a more robust and accurate forecast.
                
                This ensemble combines three model types:
                - **LSTM (Long Short-Term Memory)**: Good at learning long-term dependencies in time series data
                - **GRU (Gated Recurrent Unit)**: Similar to LSTM but simpler and sometimes trains faster
                - **Hybrid LSTM-GRU**: Combines the strengths of both architectures
                
                The final prediction is the average of all individual model predictions, which often reduces error and improves stability.
                """)
        
        with tab6:
        # --- Sentiment Analysis Section ---
            st.subheader(f"📰 News Sentiment Analysis")

            # Get user input for stock ticker and days of news to analyze
            ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")
            sentiment_days = st.slider("Days of news to analyze:", 3, 30, 7)

            if ticker_symbol:
                if st.button("Analyze News Sentiment"):
                    with st.spinner("Analyzing news sentiment... This may take a moment."):
                        try:
                            # Get sentiment data
                            news_with_sentiment, daily_sentiment = get_sentiment_feature(ticker_symbol, days=sentiment_days)
                            # DEBUG: Check columns returned
                            st.write("News with Sentiment Columns:", news_with_sentiment.columns.tolist())
                            st.write("Daily Sentiment Columns:", daily_sentiment.columns.tolist())


                            # Display the sentiment summary
                            st.success("Sentiment Analysis Complete!")

                            # Show sentiment distribution
                            sentiment_counts = news_with_sentiment['sentiment'].value_counts()
                            st.subheader("Sentiment Distribution")

                            # Create a pie chart for sentiment distribution
                            fig = px.pie(
                                names=sentiment_counts.index, 
                                values=sentiment_counts.values, 
                                title=f"Sentiment Distribution for {ticker_symbol} News",
                                color=sentiment_counts.index,
                                color_discrete_map={
                                    'Positive': 'green',
                                    'Neutral': 'gray',
                                    'Negative': 'red'
                                }
                            )
                            st.plotly_chart(fig)

                            # Show daily sentiment trend
                            st.subheader("Daily Sentiment Trend")

                            # Convert daily sentiment for plotting
                            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

                            # Create a line chart for sentiment trend
                            fig = px.line(
                                daily_sentiment, 
                                x='date', 
                                y='sentiment_score',
                                title=f"Sentiment Trend for {ticker_symbol}",
                                labels={'sentiment_score': 'Sentiment Score', 'date': 'Date'}
                            )

                            # Add reference lines
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.add_hline(y=0.05, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
                            fig.add_hline(y=-0.05, line_dash="dot", line_color="red", annotation_text="Negative Threshold")

                            st.plotly_chart(fig)

                            # Calculate average sentiment
                            avg_sentiment = news_with_sentiment['sentiment_score'].mean()
                            sentiment_status = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
                            sentiment_color = "green" if avg_sentiment > 0.05 else "red" if avg_sentiment < -0.05 else "gray"

                            st.markdown(f"<h3 style='color:{sentiment_color}'>Overall Sentiment: {sentiment_status} ({avg_sentiment:.2f})</h3>", unsafe_allow_html=True)

                            # Show correlation with price movement if possible
                            if len(daily_sentiment) > 3:
                                st.subheader("Sentiment vs. Price Movement")
                                st.write("The relationship between news sentiment and stock price movements:")

                                # Simulated price changes for demonstration
                                price_changes = np.random.randn(len(daily_sentiment))  # Example: Random price changes
                                daily_sentiment['price_change'] = price_changes

                                fig = px.scatter(
                                    daily_sentiment, 
                                    x='sentiment_score', 
                                    y='price_change',
                                    title="Sentiment Score vs Next-Day Price Change",
                                    labels={
                                        'sentiment_score': 'Sentiment Score',
                                        'price_change': 'Price Change (%)'
                                    },
                                    trendline="ols"
                                )
                                st.plotly_chart(fig)

                                # Calculate correlation
                                correlation = daily_sentiment['sentiment_score'].corr(daily_sentiment['price_change'])
                                st.write(f"Correlation between sentiment and next-day price change: **{correlation:.2f}**")

                                if abs(correlation) > 0.5:
                                    st.write("There appears to be a strong relationship between news sentiment and price movements.")
                                elif abs(correlation) > 0.3:
                                    st.write("There appears to be a moderate relationship between news sentiment and price movements.")
                                else:
                                    st.write("There appears to be a weak relationship between news sentiment and price movements.")

                            # Show news articles with sentiment
                            st.subheader("Recent News Headlines")
                            for i, row in news_with_sentiment.iterrows():
                                sentiment_color = "green" if row['sentiment'] == 'Positive' else "red" if row['sentiment'] == 'Negative' else "gray"
                                date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                                st.markdown(f"""
                                <div style="padding: 10px; margin-bottom: 10px; border-left: 5px solid {sentiment_color}; background-color: rgba(0,0,0,0.05);">
                                    <p><strong>{date_str}</strong> - {row['source']}</p>
                                    <p style="font-size: 16px;">{row['headline']}</p>
                                    <p style="color: {sentiment_color};"><strong>{row['sentiment']}</strong> ({row['sentiment_score']:.2f})</p>
                                </div>
                                """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Error in sentiment analysis: {str(e)}")
                            st.error(f"Detailed error: {type(e).__name__}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

            
            # Add explanation of sentiment analysis
            with st.expander("About Sentiment Analysis"):
                st.write("""
                Sentiment Analysis examines news headlines and articles to determine the overall market sentiment about a stock.
                
                The sentiment score ranges from -1.0 (very negative) to 1.0 (very positive):
                - **Positive**: Score > 0.05
                - **Neutral**: Score between -0.05 and 0.05
                - **Negative**: Score < -0.05
                
                News sentiment can often provide early signals about stock price movements, as positive news tends to drive prices up, while negative news can cause declines.
                """)

except Exception as e:
    st.error(f"An error occurred: {e}")
