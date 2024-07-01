import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from scipy.optimize import minimize
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go

# Funktionen für die Streamlit-App

# Portfolio-Daten erfassen (in Seitenleiste)
def get_portfolio_data():
    portfolio = []
    
    num_assets = st.sidebar.number_input("Wie viele Aktien möchten Sie hinzufügen?", min_value=1, max_value=10, value=1)
    
    for i in range(num_assets):
        ticker = st.sidebar.text_input(f"Bitte geben Sie das Aktien-Ticker-Symbol für Aktie {i+1} ein:")
        date_str = st.sidebar.date_input(f"Bitte geben Sie das Datum ein (YYYY-MM-DD), seit dem Sie investiert sind für Aktie {i+1}:", value=datetime.today() - timedelta(days=365))
        investment_amount = st.sidebar.number_input(f"Bitte geben Sie die Investmentsumme für Aktie {i+1} ein:", min_value=0.0, value=1000.0)
        
        portfolio.append({
            "ticker": ticker,
            "investment_date": date_str,
            "investment_amount": investment_amount
        })
    
    if st.sidebar.button("Daten abrufen und berechnen"):
        portfolio = fetch_historical_data(portfolio)
        total_value, portfolio_return, portfolio_volatility, current_volatility, portfolio_values = calculate_portfolio_value(portfolio)

        st.sidebar.write(f"Total Portfolio Value: {total_value}")
        st.sidebar.write(f"Portfolio Return: {portfolio_return * 100:.2f}%")
        st.sidebar.write(f"Average Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
        st.sidebar.write(f"Current Portfolio Volatility: {current_volatility * 100:.2f}%")

        expected_return, sharpe_ratio = calculate_sharpe_ratio(portfolio)

        if expected_return is not None:
            st.sidebar.write(f"Expected Return (p.a.): {expected_return * 100:.2f}%")
            st.sidebar.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        else:
            st.sidebar.write("Es gab einen Fehler bei der Berechnung der Metriken.")

        plot_portfolio_performance(portfolio_values)
        plot_asset_allocation(portfolio)

# Historische Daten abrufen
def fetch_historical_data(portfolio):
    end_date = datetime.today().strftime('%Y-%m-%d')
    for stock in portfolio:
        data = yf.download(stock['ticker'], start=stock['investment_date'].strftime('%Y-%m-%d'), end=end_date)
        stock['data'] = data
    return portfolio

# Portfolio-Wert und Performance berechnen
def calculate_portfolio_value(portfolio):
    total_investment = sum(stock['investment_amount'] for stock in portfolio)
    total_value = 0

    portfolio_values = pd.DataFrame()

    start_date = min(datetime.strptime(stock['investment_date'].strftime('%Y-%m-%d'), "%Y-%m-%d") for stock in portfolio)
    end_date = datetime.today().strftime('%Y-%m-%d')

    for stock in portfolio:
        initial_price = stock['data']['Adj Close'].iloc[0]
        current_price = stock['data']['Adj Close'].iloc[-1]
        quantity = stock['investment_amount'] / initial_price
        current_value = quantity * current_price
        stock_return = (current_price - initial_price) / initial_price

        stock['current_value'] = current_value
        total_value += current_value

        portfolio_values[stock['ticker']] = stock['data']['Adj Close'] * quantity

    portfolio_values = portfolio_values.fillna(0)
    portfolio_values['Total'] = portfolio_values.sum(axis=1)

    portfolio_return = (total_value - total_investment) / total_investment
    daily_returns = portfolio_values['Total'].pct_change().dropna()
    portfolio_volatility = np.std(daily_returns) * np.sqrt(252)

    # Berechnung der aktuellen Volatilität (letzte 30 Tage)
    recent_returns = daily_returns[-30:]
    current_volatility = np.std(recent_returns) * np.sqrt(252)

    return total_value, portfolio_return, portfolio_volatility, current_volatility, portfolio_values

# Sharpe Ratio berechnen
def calculate_sharpe_ratio(portfolio):
    tickers = [stock['ticker'] for stock in portfolio]
    investment_dates = [stock['investment_date'] for stock in portfolio]
    end_date = datetime.today()
    start_date = min(datetime.strptime(stock['investment_date'].strftime('%Y-%m-%d'), "%Y-%m-%d") for stock in portfolio)

    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        adj_close_df[ticker] = data['Adj Close']

    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    try:
        fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        risk_free_rate = ten_year_treasury_rate.iloc[-1]
    except Exception as e:
        st.sidebar.write(f"Error fetching risk-free rate: {str(e)}")
        return None

    num_assets = len(tickers)
    weights = np.array([stock['current_value'] for stock in portfolio]) / sum([stock['current_value'] for stock in portfolio])

    portfolio_expected_return = expected_return(weights, log_returns)
    portfolio_sharpe_ratio = sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    return portfolio_expected_return, portfolio_sharpe_ratio

# Grafische Darstellung der Portfolio-Performance
def plot_portfolio_performance(portfolio_values):
    fig = px.line(portfolio_values, y='Total', title='Kumulative Portfolio-Performance')
    fig.update_layout(xaxis_title='Datum', yaxis_title='Gesamtwert')
    st.plotly_chart(fig)

# Grafische Darstellung der Asset-Allokation
def plot_asset_allocation(portfolio):
    labels = [stock['ticker'] for stock in portfolio]
    sizes = [stock['current_value'] for stock in portfolio]

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3)])
    fig.update_layout(title_text='Asset Allocation')
    st.plotly_chart(fig)

# Streamlit App

# Seitenleiste für die Eingabe der Portfolio-Daten und "Berechnen" Button
st.sidebar.title("Portfolio Management App")
get_portfolio_data()






