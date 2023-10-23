import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas as pd
import yfinance as yf

# Fortune 500 companies (Sample)
fortune_500 = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]

# Add banner image
st.image("banner.PNG", use_column_width=True)

# Sidebar for input
st.sidebar.write("### Disclaimer: Assumes annualized inputs")
portfolio_value = st.sidebar.slider("Portfolio Value ($)", 1_000_000, 10_000_000, 1_000_000)
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Monthly", "Yearly"])
selected_stock = st.sidebar.selectbox("Select Company", fortune_500)

# Fetch stock data
stock_data = yf.download(selected_stock, period="10y")['Adj Close']
returns = stock_data.pct_change().dropna()

# Scaling by timeframe
timeframe_scale = {'Daily': 1, 'Monthly': 21, 'Yearly': 252}
scale_factor = np.sqrt(timeframe_scale[timeframe])
mu = returns.mean() * scale_factor
sigma = returns.std() * scale_factor

# Generate the normal distribution based on the actual returns
sample_normal = np.random.normal(mu, sigma, len(returns))
kurt_normal = stats.kurtosis(sample_normal)
kurt_actual = stats.kurtosis(returns)

# Generate distributions for plotting
x = np.linspace(min(returns) * scale_factor, max(returns) * scale_factor, 100)
y_normal = stats.norm.pdf(x, mu, sigma)
y_actual = stats.gaussian_kde(returns * scale_factor)(x)

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_normal, mode='lines', name='Normal Distribution'))
fig.add_trace(go.Scatter(x=x, y=y_actual, mode='lines', name=f'{selected_stock} Distribution'))
st.plotly_chart(fig)

# Display statistical moments
st.write("## Statistical Moments")
moments_data = {
    "Moment": ["Mean", "Variance", "Skewness", "Kurtosis"],
    "Normal": [mu, sigma ** 2, 0, kurt_normal],
    f"{selected_stock}": [mu, sigma ** 2, stats.skew(returns * scale_factor), kurt_actual]
}
st.table(pd.DataFrame(moments_data))

# VaR and CVaR calculations
percentiles = [90, 95, 99, 99.99]
var_normal = [-np.percentile(sample_normal, 100 - p) * portfolio_value for p in percentiles]
var_actual = [-np.percentile(returns * scale_factor, 100 - p) * portfolio_value for p in percentiles]
cvar_normal = [-np.mean(sample_normal[sample_normal < np.percentile(sample_normal, 100 - p)]) * portfolio_value for p in percentiles]
cvar_actual = [-np.mean(returns[returns * scale_factor < np.percentile(returns * scale_factor, 100 - p)]) * portfolio_value for p in percentiles]

# VaR and CVaR Table
var_data = {
    "Percentile": percentiles,
    "VaR Normal": var_normal,
    f"VaR {selected_stock}": var_actual,
    "CVaR Normal": cvar_normal,
    f"CVaR {selected_stock}": cvar_actual,
}
st.write("## Value at Risk and Conditional Value at Risk")
st.table(pd.DataFrame(var_data))

# Executive Summary
st.write("## Executive Summary")
st.write("""
### Definitions
- **Value at Risk (VaR)**: An upper bound on potential loss at a specific confidence level.
- **Conditional Value at Risk (CVaR)**: Averages the losses beyond the VaR threshold.

### Methodologies for Estimating VaR
1. **Historical Approach**: Directly uses historical data for calculating VaR.
2. **Variance-Covariance Method**: Assumes normal distribution of returns.
3. **Monte Carlo Method**: Simulation methodolgy to draw from multiple distributions.

### Implications in VaR and CVaR
Higher kurtosis results in underestimating risk when assuming a normal distribution. Both VaR and CVaR are sensitive to "fat tails" caused by higher kurtosis.
""")
