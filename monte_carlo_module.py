import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

# --- This function wraps your existing Monte Carlo simulation logic ---
def run_monte_carlo(Close, sample_size=100, iterations=1000, ci_level=95):
    # Calculate log returns
    log_return = np.log(Close['Close'] / Close['Close'].shift(1)).dropna()

    # Calculate drift and deviation for simulation
    u = log_return.mean()
    var = log_return.var()
    drift = u - (0.5 * var)
    deviation = log_return.std()

    # Monte Carlo Simulation
    price_list = np.zeros((sample_size, iterations))
    price_list[0] = Close['Close'].iloc[-1]

    for t in range(1, sample_size):
        random_shock = norm.ppf(np.random.rand(iterations))
        price_list[t] = price_list[t - 1] * np.exp(drift + deviation * random_shock)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(iterations):
        ax.plot(price_list[:, i], lw=0.5, alpha=0.6)

    ax.set_xlabel('Days')
    ax.set_ylabel('Simulated Price')
    ax.set_title('Monte Carlo Simulations with Brownian Motion')

    # Confidence Interval Calculation
    end_prices = price_list[-1, :]
    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower = np.percentile(end_prices, lower_percentile)
    ci_upper = np.percentile(end_prices, upper_percentile)

    return fig, (ci_lower, ci_upper),end_prices
