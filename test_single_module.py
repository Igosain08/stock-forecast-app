import pandas as pd
import yfinance as yf
import sys
import time
import numpy as np

def test_lstm_module():
    """Test just the LSTM module"""
    print("Testing LSTM module...")
    
    from lstm_module import run_lstm_forecast
    
    # Use a very small dataset for quick testing
    ticker = "AAPL"
    data = yf.download(ticker, period="6mo")
    
    print(f"Downloaded {len(data)} data points for {ticker}")
    print("Running LSTM forecast (this may take a moment)...")
    
    start_time = time.time()
    next_pred, mse, r2, mae = run_lstm_forecast(data['Close'])
    elapsed_time = time.time() - start_time
    
    print(f"LSTM test completed in {elapsed_time:.2f} seconds")
    print(f"Next day prediction: ${next_pred:.2f}")
    print(f"MSE: {mse:.6f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    
    return True

def test_monte_carlo_module():
    """Test just the Monte Carlo module"""
    print("Testing Monte Carlo module...")
    
    from monte_carlo_module import run_monte_carlo
    
    # Use a very small dataset and few iterations for quick testing
    ticker = "AAPL"
    data = yf.download(ticker, period="6mo")
    
    print(f"Downloaded {len(data)} data points for {ticker}")
    print("Running Monte Carlo simulation with minimal parameters...")
    
    start_time = time.time()
    _, (ci_lower, ci_upper), _ = run_monte_carlo(
        data,
        sample_size=5,     # Just 5 days forecast
        iterations=50,     # Only 50 iterations
        ci_level=95
    )
    elapsed_time = time.time() - start_time
    
    print(f"Monte Carlo test completed in {elapsed_time:.2f} seconds")
    print(f"95% Confidence Interval: ${ci_lower:.2f} to ${ci_upper:.2f}")
    
    return True

def test_technical_indicators():
    """Test just the technical indicators module"""
    print("Testing technical indicators module...")
    
    from technical_indicators import enrich_dataframe
    
    # Use a small dataset for quick testing
    ticker = "AAPL"
    data = yf.download(ticker, period="3mo")
    
    print(f"Downloaded {len(data)} data points for {ticker}")
    print("Calculating technical indicators...")
    
    start_time = time.time()
    enriched_data = enrich_dataframe(data)
    elapsed_time = time.time() - start_time
    
    print(f"Technical indicators test completed in {elapsed_time:.2f} seconds")
    print("Sample of calculated indicators:")
    print(enriched_data[['Close', 'SMA20', 'RSI', 'MACD']].tail())
    
    return True

def test_risk_metrics():
    """Test just the risk metrics module"""
    print("Testing risk metrics module...")
    
    from risk_metrics import calculate_returns, calculate_metrics
    
    # Use a small dataset for quick testing
    ticker = "AAPL"
    data = yf.download(ticker, period="1y")
    
    print(f"Downloaded {len(data)} data points for {ticker}")
    print("Calculating risk metrics...")
    
    start_time = time.time()
    returns = calculate_returns(data['Close'])
    metrics = calculate_metrics(returns)
    elapsed_time = time.time() - start_time
    
    print(f"Risk metrics test completed in {elapsed_time:.2f} seconds")
    print("Risk metrics results:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return True

def test_ensemble_models():
    """Test the ensemble models module"""
    print("Testing ensemble models module...")
    
    try:
        # Import the module - update the import path as needed
        from ensemble_models import create_ensemble_model, predict_next_with_ensemble
        
        # Import necessary dependencies for data preparation
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        
        # Use a small dataset for quick testing
        ticker = "AAPL"
        data = yf.download(ticker, period="6mo")
        
        print(f"Downloaded {len(data)} data points for {ticker}")
        print("Creating training data for ensemble model...")
        
        # Prepare data for the ensemble model
        # This is a simplified version of what you might have in your module
        def prepare_data(data, window_size=20):
            # Extract close prices
            close_prices = data['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            # Create training sequences
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
            
            return X_train, y_train, X_test, y_test, scaled_data, scaler
        
        # Prepare data
        window_size = 20  # Use a small window for quicker testing
        X_train, y_train, X_test, y_test, scaled_data, scaler = prepare_data(data, window_size)
        
        print(f"Data prepared. Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
        print("Creating ensemble model (this may take a moment)...")
        
        start_time = time.time()
        
        # Create the ensemble model with reduced parameters for testing
        input_shape = (X_train.shape[1], X_train.shape[2])
        ensemble = create_ensemble_model(X_train, y_train, X_test, y_test, input_shape, n_models=2)
        
        # Get the most recent window for prediction
        X_recent = scaled_data[-window_size:].reshape(1, window_size, 1)
        
        # Make a prediction
        next_pred = predict_next_with_ensemble(ensemble, X_recent)
        
        # Scale the prediction back
        next_pred_scaled = np.array([[0] * (X_train.shape[2] - 1) + [next_pred]])
        next_pred_original = scaler.inverse_transform(next_pred_scaled)[0, -1]
        
        elapsed_time = time.time() - start_time
        
        print(f"Ensemble model test completed in {elapsed_time:.2f} seconds")
        print(f"Next day prediction: ${next_pred_original:.2f}")
        print(f"Ensemble MSE: {ensemble['ensemble_mse']:.6f}")
        
        return True
    except Exception as e:
        print(f"Error testing ensemble models: {str(e)}")
        return False

def test_sentiment_analysis():
    """Test the sentiment analysis module"""
    print("Testing sentiment analysis module...")
    
    try:
        # Import the module - update the import path as needed
        from sentiment_analysis import fetch_news, analyze_sentiment, calculate_average_sentiment, get_sentiment_feature
        
        # Use a common ticker for testing
        ticker = "AAPL"
        
        print(f"Testing sentiment analysis for {ticker}")
        print("Fetching news articles (simulated)...")
        
        start_time = time.time()
        
        # Get sentiment features
        news_with_sentiment, daily_sentiment = get_sentiment_feature(ticker, days=5)
        
        elapsed_time = time.time() - start_time
        
        print(f"Sentiment analysis test completed in {elapsed_time:.2f} seconds")
        print("News articles with sentiment:")
        if 'sentiment' in news_with_sentiment.columns:
            # Display sentiment distribution
            sentiment_counts = news_with_sentiment['sentiment'].value_counts()
            print(sentiment_counts)
            
            # Display a few examples
            print("\nSample news with sentiment:")
            for i, row in news_with_sentiment.head(3).iterrows():
                print(f"Headline: {row.get('headline', 'N/A')}")
                print(f"Sentiment: {row.get('sentiment', 'N/A')} (Score: {row.get('sentiment_score', 'N/A'):.2f})")
                print("---")
                
            # Display daily sentiment
            print("\nDaily average sentiment:")
            for i, row in daily_sentiment.iterrows():
                print(f"Date: {row.get('date', 'N/A')}, Score: {row.get('sentiment_score', 'N/A'):.2f}")
        else:
            print("Sentiment data structure doesn't match expected format.")
            print(f"Available columns: {news_with_sentiment.columns.tolist()}")
        
        return True
    except Exception as e:
        print(f"Error testing sentiment analysis: {str(e)}")
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Please specify which module to test:")
        print("1. lstm - Test LSTM forecasting module")
        print("2. monte_carlo - Test Monte Carlo simulation module")
        print("3. technical - Test technical indicators module")
        print("4. risk - Test risk metrics module")
        print("5. ensemble - Test ensemble models module")
        print("6. sentiment - Test sentiment analysis module")
        sys.exit(1)
    
    module = sys.argv[1].lower()
    
    if module == "lstm":
        test_lstm_module()
    elif module == "monte_carlo":
        test_monte_carlo_module()
    elif module == "technical":
        test_technical_indicators()
    elif module == "risk":
        test_risk_metrics()
    elif module == "ensemble":
        test_ensemble_models()
    elif module == "sentiment":
        test_sentiment_analysis()
    else:
        print(f"Unknown module: {module}")
        print("Available modules: lstm, monte_carlo, technical, risk, ensemble, sentiment")
