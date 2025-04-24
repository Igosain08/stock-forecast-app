
# 📊 MarketScope: Intelligent Stock Forecasting App

**MarketScope** is a powerful Streamlit-based application for forecasting stock prices. It integrates deep learning (LSTM, GRU, Hybrid), ensemble models, Monte Carlo simulations, technical indicators, risk metrics, and sentiment analysis of financial news to provide a comprehensive market insight tool.

---

## 👨‍💻 Made By

- **Vedant Vardhaan**  
- **Ishaan Gosain**

---

## 🚀 Features

- 📈 **Stock Price Forecasting** using LSTM, GRU, and Hybrid models with metrics (MSE, R², MAE).
- 🤖 **Ensemble Forecasting** that combines predictions from multiple models for better accuracy.
- 🔮 **Monte Carlo Simulation** to estimate future price distribution and volatility.
- 📊 **Technical Indicators** including Moving Averages, RSI, MACD, Bollinger Bands.
- 🧠 **Risk Analysis** with statistical summaries and visualizations.
- 📰 **News Sentiment Analysis** to correlate market mood with stock price trends.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/marketscope.git
   cd marketscope
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Models Implemented

- **LSTM**: Long Short-Term Memory Neural Network
- **GRU**: Gated Recurrent Unit Model
- **Hybrid**: Combination of LSTM and GRU
- **Ensemble**: Aggregated predictions from LSTM, GRU, and Hybrid

---

## 📦 Project Structure

```
marketscope/
├── app.py                    # Main Streamlit frontend
├── lstm_module.py           # LSTM forecasting logic
├── monte_carlo_module.py    # Monte Carlo simulations
├── technical_indicators.py  # Indicator computation
├── risk_metrics.py          # Risk evaluation functions
├── model_utils.py           # GRU, Hybrid model definitions
├── ensemble_models.py       # Ensemble learning logic
├── sentiment_analysis.py    # Financial news sentiment extraction
├── requirements.txt         # Project dependencies
```

---

## 📈 Data Sources

- Historical stock prices via [Yahoo Finance](https://finance.yahoo.com/)
- News headlines from financial news APIs (handled in `sentiment_analysis.py`)

---

## 📊 Visuals and Insights

- Interactive time-series charts using Plotly
- Predicted vs. actual performance graphs
- Confidence intervals and volatility bands
- Daily sentiment trends with correlation to price changes

---

## ✨ Future Enhancements

- Live market data streaming
- Portfolio-level analysis and forecasting
- Sentiment-driven trading alerts
- Improved dashboard and user interface

---

## 🙌 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
