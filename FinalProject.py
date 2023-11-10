# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:38:28 2023

@author: lmartinez1
"""

# Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
    

#==============================================================================
# Tab 1 - Summary
#==============================================================================
# Get the list of stock tickers from S&P500
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Define the main function for Tab 1
def tab1():
    st.header('Summary')
    
    # Divide the screen into three columns
    col1, col2, col3 = st.columns([3, 4, 3])
    
    # Cache function to retrieve company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        return YFinance(ticker).info

    # Check the selection of a ticker
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)

        # Display the business summary of the company
        st.write('**1. Business Summary:**')
        st.markdown(
            '<div style="text-align: justify;">' + info['longBusinessSummary'] + '</div><br>',
            unsafe_allow_html=True
        )

        # Display key statistics of the company
        st.write('*Important Statistics*')    
        # Create dictionaries for stats in col1 and col2
        stats_col1 = {
            'Previous Close': info.get('previousClose'),
            'Open': info.get('open'),
            'Bid': info.get('bid'),
            'Ask': info.get('ask'),
            'Day Range': info.get('daysrange'),
            '52 Week Range': info.get('52weekrange'),
            'Volume': info.get('volume'),
            'Avg. Volume': np.mean(info.get('volume'))
        }
        
        stats_col2 = {
            'Market Cap': info.get('marketCap'),
            'Beta (5Y Monthly)': info.get('beta (5Y Monthly)'),
            'PE Ratio (TTM)': info.get('peRatio (ttm)'),
            'EPS (TTM)': info.get('eps (ttm)'),
            'Earnings Date': info.get('earningsdate'),
            'Forward Dividend & Yield': info.get('forwarddividendandyield'),
            'EX-Dividend Date': pd.to_datetime(info.get('exDividendDate'), unit = 's'),
            '1Y Target Est': info.get('1ytargetest')
        }
        
        # Display statistics in DataFrame format in respective columns
        with col1:
            # Convert to DataFrame
            company_stats_col1 = pd.DataFrame({'Value': pd.Series(stats_col1)})
            st.dataframe(company_stats_col1)
        
        with col2:
            # Convert to DataFrame
            company_stats_col2 = pd.DataFrame({'Value': pd.Series(stats_col2)})
            st.dataframe(company_stats_col2)
        
        # Retrieve stock data for the selected period
        stock_data =yf.download(tickers = ticker,period = "3y")
         
        # Display buttons to select different time periods
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1: date1 = st.button('1M')
        with col2: date2 = st.button('3M')
        with col3: date3 = st.button('6M')
        with col4: date4 = st.button('YTD')
        with col5: date5 = st.button('1Y')
        with col6: date6 = st.button('3Y')
        with col7: date7 = st.button('5Y')
        with col8: date8 = st.button('MAX')
        
        # Check the selected button and show data accordingly
        if date1:
            stock_data=yf.download(tickers = ticker,period = "1mo")
        if date2:
            stock_data=yf.download(tickers = ticker,period = "3mo")
        if date3:
            stock_data=yf.download(tickers = ticker,period = "6mo")
        if date4:
            stock_data=yf.download(tickers = ticker,period = "ytd")
        if date5:
            stock_data=yf.download(tickers = ticker,period = "1y")
        if date6:
            stock_data=yf.download(tickers = ticker,period = "3y")
        if date7:
             stock_data=yf.download(tickers = ticker,period = "5y")
        if date8:
            stock_data=yf.download(tickers = ticker)
         
        # Plot the stock price data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], fill='tozeroy',  name=ticker))

        fig.update_layout(
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            )
            
        st.plotly_chart(fig, use_container_width=True)
        
#==============================================================================
# Tab 2 - Chart
#==============================================================================

# Define the main function for Tab 2
def tab2():
    st.title("Chart")
    
    # Divide the screen into three columns
    c1, c2, c3 = st.columns((1, 1, 1))
    
    # Create boxes to select duration, interval, and plot type
    with c1:
        duration = st.selectbox("Select duration", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'])
    with c2:
        inter = st.selectbox("Select interval", ['1d', '1mo', '1Y'])
    with c3:
        plot = st.selectbox("Select Plot", ['Line', 'Candle Plot'])

    # Fetch Stock Data using the global `ticker` variable
    end_date = pd.Timestamp.now()
    
    # Determine start date based on the selected duration
    if duration == '1M':
        start_date = end_date - pd.DateOffset(months=1)
    elif duration == '3M':
        start_date = end_date - pd.DateOffset(months=3)
    elif duration == '6M':
        start_date = end_date - pd.DateOffset(months=6)
    elif duration == 'YTD':
        start_date = pd.Timestamp(datetime(end_date.year, 1, 1))
    elif duration == '1Y':
        start_date = end_date - pd.DateOffset(years=1)
    elif duration == 'MAX':  
        start_date = end_date - pd.DateOffset(years=50)
    else:
        start_date = end_date - pd.DateOffset(years=int(duration[:-1]))

    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=inter)

    # Scale the volume data
    max_price = stock_data['Close'].max()
    max_volume = stock_data['Volume'].max()
    scale_factor = max_price / max_volume
    scaled_volume = stock_data['Volume'] * scale_factor

    # Create a subplot with dual y-axes
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Add traces based on the selected plot type
    if plot == 'Candle Plot':
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'], high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], name='Candlesticks'))
        # Add theLine plot
    else:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='green')))

    # Add volume as a bar-plot on a secondary y-axis
    fig.add_trace(go.Bar(x=stock_data.index, y=scaled_volume, name='Volume'), secondary_y=False)

    # Set y-axis titles
    fig.update_yaxes(title_text='Price', secondary_y=False)
    fig.update_yaxes(title_text='Scaled Volume', secondary_y=True)

    # Set titles
    fig.update_layout(
        title=f'{ticker} Stock Price Chart',
        xaxis_rangeslider_visible=True,
        xaxis_title='Date',
    )

    # Display the chart
    st.plotly_chart(fig)
           
#==============================================================================
# Tab 3 - Financials
#==============================================================================

# Define the main function for Tab 3
def tab3():
    st.title('Financials')
    
    # Dropdown box to select simulations and time horizon
    statement = st.selectbox("Choose intervals:", ['Yearly', 'Quarterly'])   
    
    # Divide the screen into four columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Create buttons to select different financial statements
    Op1 = col1.button('Income Statement') 
    Op2 = col2.button('Balance Sheet')
    Op3 = col3.button('Cash Flow')
    
    # Initialize a variable to display the content
    stock = yf.Ticker(ticker)
    
    display_default = True
    
    # Display income_statement based on button selection and interval choice
    if Op1:
        display_default = False
        if statement == 'Quarterly':
            income_statement = stock.quarterly_financials
            st.write(income_statement)
        else:
            income_statement = stock.financials
            st.write(income_statement)
    
    # Display balance_sheet based on button selection and interval choice
    if Op2:
        if display_default:
            display_default = False
        if statement == 'Quarterly':
            balance_sheet = stock.quarterly_balancesheet
            st.write(balance_sheet)
        else:
            balance_sheet = stock.balancesheet
            st.write(balance_sheet)
    
    # Display cash_flow statement based on button selection and interval choice
    if Op3:
        if display_default:
            display_default = False
        if statement == 'Quarterly':
            cash_flow = stock.quarterly_cashflow
            st.write(cash_flow)
        else:
            cash_flow = stock.cashflow
            st.write(cash_flow)
    
#==============================================================================
# Tab 4 - Monte Carlo Simulation
#==============================================================================

# Define the main function for Tab 4
def tab4():
    st.title('Monte Carlo Simulation')
    
    # Dropdown box to select simulations and time horizon
    simulations = st.selectbox("Select number of Simulations:", [200, 500, 1000], index=1)
    time_horizon = st.selectbox("Select time horizon:", [30, 60, 90], index=1)

    # Get stock data
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period="max")

    # Calculate daily returns and volatility
    close_price = hist_data['Close']
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)

    # Set the random seed for reproducibility
    np.random.seed(123)

    # Use a dataframe to store simulation data
    simulation_df = pd.DataFrame()

    # Monte Carlo simulation
    for i in range(simulations):
        next_price = []
        last_price = close_price[-1]

        for j in range(time_horizon):
            # Generate a random future_return based volatility previously calculated
            future_return = np.random.normal(0, daily_volatility)
            future_price = last_price * (1 + future_return)
            next_price.append(future_price)
            last_price = future_price

        simulation_df[i] = next_price

    # Plot the MC simulation
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)

    plt.plot(simulation_df)
    plt.title(f"Monte Carlo simulation for {ticker} stock price in next {time_horizon} days")
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.axhline(y=close_price[-1], color='red')
    plt.legend([f'Current stock price is: {np.round(close_price[-1], 2)}'])
    ax.get_legend().legendHandles[0].set_color('red')

    #Display de plot
    st.pyplot(plt)

    # Calculate VaR at 95% of confidence
    ending_price = simulation_df.iloc[-1, :]
    future_price_95ci = np.percentile(ending_price, 5)
    VaR = close_price[-1] - future_price_95ci

    #Display VaR
    st.subheader(f'VaR at 95% confidence interval is: {np.round(VaR, 2)} USD')
 
#==============================================================================
# Tab 5 - Analysis
#==============================================================================       

# Define the main function for Tab 5
def tab5():
    st.title('Analysis')
    
    # Add multiselect button to choose tickers and compare them
    selected_tickers = st.multiselect("Select tickers", ticker_list)

    if selected_tickers:
        data = yf.download(selected_tickers, period='5y')['Close']

        if isinstance(data, pd.DataFrame):
            fig = go.Figure()
            for col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))
        else:  # When data is a pandas Series for a single ticker
            fig = go.Figure(data=go.Scatter(x=data.index, y=data.values, mode='lines', name=selected_tickers[0]))

        fig.update_layout(title="Stock Price Trend Comparison",
                          xaxis_title="Date",
                          yaxis_title="Price")
        
        st.plotly_chart(fig)

#==============================================================================
#HEADER
#==============================================================================
      
# Define the header
def run():
    # Set title and additional information
    st.sidebar.title("Financial Dashboard")
    st.sidebar.write("Data source: Yahoo Finance")
    
    # Add the ticker selection on the sidebar as a box
    global ticker
    ticker = st.sidebar.selectbox("Select ticker", ticker_list)
    
    # Add a button to update the data based on ticker selection   
    run_button = st.sidebar.button('Update Data')
    if run_button:
        st.experimental_rerun()
    
    # Add a radio box to select different tabs
    select_tab = st.sidebar.radio("Select tab", ['Summary','Chart','Financials','Monte Carlo Simulation','Personal Analysis'])
    
    # Show the selected tab
    if select_tab == 'Summary':
        # Run tab 1 - Summary
        tab1()
    elif select_tab == 'Chart':
        # Run tab 2 - Chart
        tab2()
    elif select_tab == 'Financials':
        # Run tab 3 - Financials
        tab3()
    elif select_tab == 'Monte Carlo Simulation':
        # Run tab 4 - MC Simulation
        tab4()
    elif select_tab == 'Personal Analysis':
        # Run tab 5 - Personal Analysis
        tab5()

        
if __name__ == "__main__":
    run()    


###############################################################################
# END
#############################################