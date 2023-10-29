import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas as pd
import yfinance as yf

st.title("The Impact of Excess Kurtosis on Value at Risk and Expected Shortfall")

# Sidebar for input
st.sidebar.write("***Disclaimer: using up to 10y of historical data")
st.sidebar.title("Parameters")
portfolio_value = st.sidebar.slider("Portfolio Value ($)", 1_000_000, 10_000_000, 1_000_000)
timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "Monthly", "Yearly"])
selected_stock = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
st.sidebar.title("A BytePotion App")
st.sidebar.image("bytepotion.png", width=200)  # Replace with your image URL or file path
st.sidebar.write("This app provides insight into the impact of excess kurtosis on VaR and CVaR for selected Fortune 500 companies.")
st.sidebar.write("https://bytepotion.com")
st.sidebar.title("Author")
st.sidebar.image("roman2.png", width=200)  # Replace with your image URL or file path
st.sidebar.write("Roman Paolucci")
st.sidebar.write("MSOR Graduate Student @ Columbia University")
st.sidebar.write("roman.paolucci@columbia.edu")

# Fetch stock data
stock_data = yf.download(selected_stock, period="10y")['Adj Close']

if stock_data.empty:
    st.error("No Data for Given Ticker")
else:
    returns = stock_data.pct_change().dropna()
    
    # Scaling by timeframe
    timeframe_scale = {'Daily': 1, 'Monthly': 21, 'Yearly': 252}
    scale_factor = np.sqrt(timeframe_scale[timeframe])
    mu = returns.mean() * scale_factor
    sigma = returns.std() * scale_factor

    # Generate the normal distribution based on the actual returns
    sample_normal = np.random.normal(mu, sigma, len(returns))

    # Generate distributions for plotting
    x = np.linspace(min(returns) * scale_factor, max(returns) * scale_factor, 100)
    y_normal = stats.norm.pdf(x, mu, sigma)
    y_actual = stats.gaussian_kde(returns * scale_factor)(x)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_normal, mode='lines', name='Normal Distribution'))
    fig.add_trace(go.Scatter(x=x, y=y_actual, mode='lines', name=f'{selected_stock} Distribution'))
    st.plotly_chart(fig)

    # VaR and CVaR calculations
    percentiles = [90, 95, 99, 99.99]
    var_normal = [-np.percentile(sample_normal, 100 - p) * portfolio_value for p in percentiles]
    var_actual = [-np.percentile(returns * scale_factor, 100 - p) * portfolio_value for p in percentiles]
    var_diff = [vn - va for vn, va in zip(var_normal, var_actual)]

    cvar_normal = [-np.mean(sample_normal[sample_normal < np.percentile(sample_normal, 100 - p)]) * portfolio_value for p in percentiles]
    cvar_actual = [-np.mean(returns[returns * scale_factor < np.percentile(returns * scale_factor, 100 - p)]) * portfolio_value for p in percentiles]
    cvar_diff = [cn - ca for cn, ca in zip(cvar_normal, cvar_actual)]

    # VaR Table
    var_data = {
        "Percentile": percentiles,
        "VaR Normal": var_normal,
        f"VaR {selected_stock}": var_actual,
        "Difference": var_diff
    }
    st.write("## Value at Risk (VaR)")
    st.table(pd.DataFrame(var_data))

    # CVaR Table
    cvar_data = {
        "Percentile": percentiles,
        "CVaR Normal": cvar_normal,
        f"CVaR {selected_stock}": cvar_actual,
        "Difference": cvar_diff
    }
    st.write("## Conditional Value at Risk (CVaR)")
    st.table(pd.DataFrame(cvar_data))

    # Executive Summary in Streamlit
    st.write("## Executive Summary")

    st.write("""
    ### Different Methods for Calculating VaR and CVaR

    1. **Historical Method**: 
        - **Description**: Directly uses historical data to estimate VaR and CVaR values.
        - **Pros**: Simple to implement.
        - **Cons**: Assumes past performance is indicative of future risk.

    2. **Variance-Covariance Method (Parametric Method)**: 
        - **Description**: Assumes that returns follow a normal distribution. Utilizes the mean and standard deviation of past returns for calculations.
        - **Pros**: Computationally efficient.
        - **Cons**: Inaccurate if returns are not normally distributed.

    3. **Monte Carlo Simulation**: 
        - **Description**: Utilizes simulated data to model potential future scenarios. Useful for complex financial instruments.
        - **Pros**: Comprehensive and adaptable.
        - **Cons**: Computationally expensive.

    ### Implications of Excess Kurtosis on VaR and CVaR

    - **Excess Kurtosis**: Describes the "tailedness" of the distribution. Positive excess kurtosis indicates "fat tails."

    - **Impact on VaR and CVaR**: 
        - A distribution with higher kurtosis (fat tails) will underestimate risk when assuming normality.
        - Both VaR and CVaR values calculated under the assumption of normality will be too optimistic, potentially leading to insufficient risk coverage.

    - **Problems with Assuming Normality**:
        - Ignoring kurtosis leads to underestimating the likelihood of extreme losses.
        - For instruments or portfolios subject to extreme market events, assuming normality can be highly misleading.
    """)

