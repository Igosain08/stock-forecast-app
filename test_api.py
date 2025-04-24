from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# === Test Connection ===
@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({'status': 'success', 'message': 'API is up and running!'})

from datetime import datetime

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    try:
        # Fetch stock data using yfinance
        data = yf.download(ticker, period="1mo")

        # Prepare the prices list
        prices = []
        for date, price in data['Close'].items():
            # If the date is a string, convert it to datetime object
            if isinstance(date, str):
                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")  # Convert string to datetime object
            
            prices.append({'date': date.strftime("%Y-%m-%d %H:%M:%S"), 'price': float(price)})

        return jsonify({'status': 'success', 'ticker': ticker, 'prices': prices})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# === Ensemble Forecast ===
@app.route('/api/ensemble-forecast', methods=['POST'])
def ensemble_forecast():
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        ticker = req_data.get('ticker', 'AAPL')
        window_size = int(req_data.get('window_size', 20))
        prediction_days = int(req_data.get('prediction_days', 5))

        data = yf.download(ticker, period="1y")
        if data.empty:
            return jsonify({'status': 'error', 'message': f'No data found for {ticker}'}), 404

        from ensemble_models import create_ensemble_model, predict_next_with_ensemble
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        def prepare_data(data, window_size=20):
            close_prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            X, y = [], []
            for i in range(window_size, len(scaled_data)):
                X.append(scaled_data[i-window_size:i, 0])
                y.append(scaled_data[i, 0])
            X = np.array(X)
            y = np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            return X_train, y_train, X_test, y_test, scaled_data, scaler

        X_train, y_train, X_test, y_test, scaled_data, scaler = prepare_data(data, window_size)
        input_shape = (X_train.shape[1], X_train.shape[2])
        ensemble = create_ensemble_model(X_train, y_train, X_test, y_test, input_shape)

        X_recent = scaled_data[-window_size:].reshape(1, window_size, 1)
        next_pred = predict_next_with_ensemble(ensemble, X_recent)
        next_pred_scaled = np.array([[0] * (X_train.shape[2] - 1) + [next_pred]])
        next_pred_original = scaler.inverse_transform(next_pred_scaled)[0, -1]

        historical = [{'date': date.strftime('%Y-%m-%d'), 'price': float(close)} for date, close in data['Close'][-30:].items()]
        predictions = []
        last_date = data.index[-1]

        for i in range(1, prediction_days + 1):
            future_date = last_date + timedelta(days=i)
            while future_date.weekday() > 4:
                future_date += timedelta(days=1)
            predicted_price = next_pred_original * (1 + 0.002 * i)
            predictions.append({'date': future_date.strftime('%Y-%m-%d'), 'price': float(predicted_price)})

        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'historical': historical,
            'predictions': predictions,
            'ensemble_mse': float(ensemble['ensemble_mse'])
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# === Forecast Endpoint Placeholder ===
@app.route('/api/forecast', methods=['POST'])
def forecast_stock():
    req_data = request.get_json()
    return jsonify({'status': 'success', 'message': 'Forecast placeholder response', 'input': req_data})

# === Monte Carlo Simulation Placeholder ===
@app.route('/api/monte-carlo', methods=['POST'])
def monte_carlo_simulation():
    req_data = request.get_json()
    return jsonify({'status': 'success', 'message': 'Monte Carlo simulation placeholder', 'input': req_data})

# === Technical Indicators Placeholder ===
@app.route('/api/technical', methods=['GET'])
def get_technical_indicators():
    ticker = request.args.get('ticker')
    start = request.args.get('start')
    end = request.args.get('end')
    return jsonify({'status': 'success', 'message': 'Technical indicators placeholder', 'ticker': ticker, 'start': start, 'end': end})

# === Risk Metrics Placeholder ===
@app.route('/api/risk-metrics', methods=['GET'])
def get_risk_metrics():
    ticker = request.args.get('ticker')
    period = request.args.get('period')
    return jsonify({'status': 'success', 'message': 'Risk metrics placeholder', 'ticker': ticker, 'period': period})

# === Sentiment Analysis ===
@app.route('/api/sentiment/<ticker>', methods=['GET'])
def get_sentiment(ticker):
    try:
        from sentiment_analysis import get_sentiment_feature
        days = int(request.args.get('days', 7))
        news_with_sentiment, daily_sentiment = get_sentiment_feature(ticker, days=days)

        news_list = [{
            'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
            'headline': row['headline'],
            'source': row['source'],
            'sentiment': row['sentiment'],
            'sentiment_score': float(row['sentiment_score'])
        } for _, row in news_with_sentiment.iterrows()]

        daily_list = [{
            'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
            'sentiment_score': float(row['sentiment_score'])
        } for _, row in daily_sentiment.iterrows()]

        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'news': news_list,
            'daily_sentiment': daily_list,
            'average_sentiment': float(daily_sentiment['sentiment_score'].mean())
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# === Market Regime Placeholder ===
@app.route('/api/market-regime', methods=['GET'])
def detect_market_regime():
    ticker = request.args.get('ticker')
    window = request.args.get('window')
    return jsonify({'status': 'success', 'message': 'Market regime detection placeholder', 'ticker': ticker, 'window': window})

# === Start Server ===
if __name__ == "__main__":
    print("Starting Flask API...")
    app.run(debug=True)
