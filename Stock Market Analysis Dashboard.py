#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas_datareader.data as wb
import yfinance as yf
yf.pdr_override()
import sqlite3
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')

# NYT API ************************************************

from pynytimes import NYTAPI

import os
import json
import time
import requests
import datetime as dt
from datetime import date 
import dateutil
from dateutil.relativedelta import relativedelta
apikey = open(r"C:\Users\David_King\Downloads\Stock_Market_Analysis_Tool_Project\NYT_API_KEY.txt", "r")
nyt = NYTAPI(apikey.read(), parse_dates=True, backoff=True)


from ratelimit import limits, RateLimitException, sleep_and_retry
from textblob import TextBlob

# PLOTLY/DASH *********************************************

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# In[2]:


# Downloading current companies on NASDAQ

payload=pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
fifth_table = payload[4]
NASDAQ_Ticker = fifth_table
NASDAQ_Ticker['stock_symbols'] = NASDAQ_Ticker['Ticker'] + ' ' + NASDAQ_Ticker['Company']
NASDAQ_Ticker = NASDAQ_Ticker[['stock_symbols', 'Company', 'Ticker', 'GICS Sector', 'GICS Sub-Industry']]
EMA_Dic = {'8 Day EMA':8, '20 Day EMA':20, '50 Day EMA':50, '200 Day EMA':200}


# Note: 
# 
# 'response' is a dictionary object; within the function, work to convert it into a dataframe and 
# drop unncessary columns, do sentiment anaylsis on the abstract for each article, and make a tuple of the
# artcle title, sentiment analysis, and article abstract
# 

# In[3]:


options = {
    "sort":"relevance",
    "news_desk":["Business Day",
                 "Business",
                 "DealBook",
                 "Financial",
                 "Market Place",
                 "Technology",
                 "U.S."],
    "type_of_material": ["News Analysis", 
                         "News", 
                         "Article"]}
@sleep_and_retry
@limits(calls=10, period=60, raise_on_limit = True)
def fetch_articles(company, article_dict, start_date, end_date):
    

    try:
        response = nyt.article_search(query=company, results=10, dates={'begin':start_date, 'end':end_date}, options=options)
        
        if company not in article_dict:
            article_dict[company] = response
        else:
            article_dict[company].append(response)                 
            return article_dict
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# In[4]:


def article_dic_to_df(df, article_dict, company):
    temp_df = pd.DataFrame.from_records(article_dict[company])
    temp_df.columns = pd.MultiIndex.from_product([[company], temp_df.columns])
    return pd.concat([df, temp_df], axis = 1)


# In[5]:


def article_abstract_sentiment(df, company): 
    df[(company, "sentiment")] = df[(company, "abstract")].apply(lambda x: TextBlob(x))
    df[(company, "sentiment")] = df[(company, "sentiment")].map(lambda x: (x.sentiment.polarity))
    return df


# In[6]:


def company_to_ticker_index(df):
    
    company_name = list(df.columns.levels[0])
    ticker_list = []
    for item in company_name:
        
        # Update 'pub_date' columnn from timestamp to date
        
        df[(f'{item}', 'pub_date')] = df[(f'{item}', 'pub_date')].apply(lambda x: x.date()) 
        
        # Update the headline to extract the value for 'main' from the dict
        
        df[f'{item}', 'headline'] = df[f'{item}', 'headline'].apply(lambda x: x.get('main') if isinstance(x, dict) else x)
        
        # Find the corresponding ticker symbol for the company
        ticker = NASDAQ_Ticker.loc[NASDAQ_Ticker['Company'] == item]['Ticker'].iloc[0]
        ticker_list.append(ticker)
    
    df.columns = df.columns.set_levels(ticker_list, level=0)
    
    
    return df


# # Stock Dashboard

# In[7]:


app = Dash(__name__)


# In[8]:


app.layout = html.Div([
    html.H1('Stock Ticker Dashboard'),
    
    html.Div([
        
        html.H3('Enter a stock symbol:', style={'paddingRight':'30px'}),
        dcc.Dropdown(
            options =[{'label': i, 'value': i} for i in NASDAQ_Ticker['stock_symbols']],
            value = ['TSLA Tesla, Inc.'],
            multi = True, clearable = True, 
            searchable = True, id='demo_dropdown'),
        html.H3('Select an analysis type:', style={'paddingRight':'30px'}),
        dcc.Dropdown(
            options =[{'label':'Closing Price', 'value': 'Closing Price'},
                      {'label':'Bollinger Bands', 'value': 'Bollinger Bands'},
                      {'label':'Volumetric Moving Average (VMA)', 'value': 'Volumetric Moving Average (VMA)'},
                      {'label':'Relative Strength Index(RSI)', 'value': 'Relative Strength Index(RSI)'}
                     ],
            value = 'Closing Price',
            multi=True, clearable=True,
            id='analysis_type_dropdown')],            
        style={'display':'inline-block', 'verticalAlign':'top', 'width':'30%'}),
    
    html.H3('Select start and end dates:'),
    
    html.Div([
    dcc.DatePickerRange(
        id='date_range',
        min_date_allowed = date(1962, 1, 1),
        max_date_allowed = dt.date.today(),
        start_date = date(2023, 1, 1),
        end_date = dt.date.today(),
        updatemode = 'singledate')], 
        style={'display':'inline-block'}),
            
    html.Div([
    html.Button(
        id='submit_button',
        n_clicks = 0,
        children='Submit',
        style={'fontSize':24, 'marginLeft':'30px'})],
        style={'display':'inline-block'}),
    
    
    html.Div([
       dcc.Checklist(
        id='Exponential_Moving_Average_Checkbox',
        options=[{'label': key, 'value': value} for key, value in EMA_Dic.items()],
        value=[]),
        dcc.Checklist(
        id='Candlestick_Graph',
        options=[{'label': 'Candlestick', 'value':'Candlestick'}],
        value=[])]),
    
    html.Div([
        dcc.Graph(id='feature_graphic')])
    
    ])
    


# In[9]:


@app.callback(Output('feature_graphic', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('demo_dropdown', 'value'),
     State('date_range', 'start_date'),
     State('date_range', 'end_date'),
     State('analysis_type_dropdown', 'value'),
     State('Exponential_Moving_Average_Checkbox', 'value'),
     State('Candlestick_Graph', 'value')
    ]) 
    
def update_output(n_clicks, dropdown_value, start_date, end_date, analysis_type_dropdown, EMA_checkbox_value, Candlestick_Graph_Value):
    row = {}
    graph_data = []
    VMA_subplot_data = []
    RSI_subplot_data = []
    data_dic = {}
    article_df = pd.DataFrame()
    article_dict = {}

  
    if not dropdown_value:
        return dash.no_update    
    
    if dropdown_value:
        ticker_list = NASDAQ_Ticker.loc[NASDAQ_Ticker['stock_symbols']\
                                       .isin(dropdown_value)]['Ticker'].tolist() 

        company_list = sorted(NASDAQ_Ticker.loc[NASDAQ_Ticker['stock_symbols']\
                                       .isin(dropdown_value)]['Company'].tolist())     
        for ticker in ticker_list:
            if ticker not in data_dic:
                df = wb.DataReader(ticker, start = start_date, end = end_date)
                df.index = [x.date() for x in df.index.normalize().to_pydatetime()]
                data_dic[ticker] = df

# Closing Price *********************************************************************************

        if 'Closing Price' in analysis_type_dropdown:
            row['close_scatter'] = 1
            for ticker in ticker_list:
                close_scatter = go.Scatter(
                    x=data_dic[ticker].index,
                    y=data_dic[ticker]['Close'],
                    mode='lines',
                    text=ticker,
                    name=ticker  # set name to display in legend
                )
                graph_data.append(close_scatter)

# EMA *********************************************************************************
        if EMA_checkbox_value:  # If the checkbox has a value (is checked)

            for value in EMA_checkbox_value:
                for ticker in ticker_list:
                    EMA_value = ta.EMA(data_dic[ticker]['Close'], timeperiod = value)
                    scatter = go.Scatter(
                        x=EMA_value.index,
                        y=EMA_value,
                        mode='lines',
                        text=ticker,
                        name=f"{ticker}_{value}_Day_EMA "
                    )
                    graph_data.append(scatter)

# Candlestick **********************************************************************************
    
        if Candlestick_Graph_Value:  # If the checkbox has a value (is checked)

                for ticker in ticker_list:
                    Candlestick = go.Candlestick(x=data_dic[f"{ticker}"].index,
                                    open=data_dic[f"{ticker}"]['Open'],
                                    high=data_dic[f"{ticker}"]['High'],
                                    low=data_dic[f"{ticker}"]['Low'], 
                                    close=data_dic[f"{ticker}"]['Close'])

                    graph_data.append(Candlestick)

# BB *************************************************************************************

        if 'Bollinger Bands' in analysis_type_dropdown:

            BB_Dic = {}
            for ticker in ticker_list:
                BB_Dic[f'{ticker}_BB_U'],  BB_Dic[f'{ticker}_BB_SMA'], BB_Dic[f'{ticker}_BB_L'] = \
                ta.BBANDS(data_dic[ticker]['Close'],timeperiod=20, nbdevup=2,nbdevdn=2,matype=0)

                BB_U_scatter = go.Scatter(
                    x=BB_Dic[f'{ticker}_BB_U'].index,
                    y=BB_Dic[f'{ticker}_BB_U'].values,
                    mode='lines',
                    line=dict(color='rgb(203,213,232)'),
                    text=ticker,
                    name="Upper Bollinger Band"
                )
                graph_data.append(BB_U_scatter)

                BB_L_scatter = go.Scatter(
                    x=BB_Dic[f'{ticker}_BB_L'].index,
                    y=BB_Dic[f'{ticker}_BB_L'].values,
                    mode='lines',
                    line=dict(color='rgb(203,213,232)'),
                    fill='tonexty',
                    text=ticker,
                    name="Upper Bollinger Band")

                graph_data.append(BB_L_scatter)

                BB_SMA_scatter = go.Scatter(
                    x=BB_Dic[f'{ticker}_BB_SMA'].index,
                    y=BB_Dic[f'{ticker}_BB_SMA'].values,
                    mode='lines',
                    line=dict(color='grey'),
                    text=ticker,
                    name="Moving Average")

                graph_data.append(BB_SMA_scatter)
    
# VMA ***************************************************************************************

        if 'Volumetric Moving Average (VMA)' in analysis_type_dropdown:

            volume_dict = {}

            for ticker in ticker_list:

                volume_dict[f'{ticker}'] = data_dic[f'{ticker}']['Volume'].reset_index()
                volume_dict[f'{ticker}']['VMA'] = ta.SMA(volume_dict[f'{ticker}']['Volume'], timeperiod = 20)
                volume_dict[f'{ticker}'].set_index('index', inplace=True)


                VMA_scatter = go.Scatter(
                    x=volume_dict[f'{ticker}'].index,
                    y=volume_dict[f'{ticker}']['VMA']/100000000,
                    mode='lines',
                    text=ticker,
                    name=f"{ticker}_SMA"
                )

                volume_bar = go.Bar(x=volume_dict[f'{ticker}'].index,
                                    y=volume_dict[f'{ticker}']['Volume']/100000000,
                                    base="group", 
                                    name="Trading Volume")

                VMA_subplot_data.append(VMA_scatter)
                VMA_subplot_data.append(volume_bar)

# RSI ***************************************************************************************

#         AAPL_data['RSI'] = ta.RSI(AAPL_data['Close'], timeperiod=14)

#         ## Plot RSI subsplot
#         ax2.title.set_text('Relative Strength Indicator (RSI)')
#         ax2.plot(AAPL_data['RSI'], color='tab:gray')
#         plt.axhline(y=70, color='tab:blue', linestyle="dashed")
#         plt.axhline(y=30, color='tab:orange', linestyle="dashed")
    
        if 'Relative Strength Index(RSI)' in analysis_type_dropdown:

            RSI_dict = {}

            for ticker in ticker_list:

                RSI_dict[f'{ticker}'] = data_dic[f'{ticker}']['Close'].reset_index()
                RSI_dict[f'{ticker}']['RSI'] = ta.RSI(RSI_dict[f'{ticker}']['Close'], timeperiod = 14)
                RSI_dict[f'{ticker}'].set_index('index', inplace=True)

                x_values = RSI_dict[f'{ticker}'].index
                RSI_scatter = go.Scatter(
                    x=x_values,
                    y=RSI_dict[f'{ticker}']['RSI'],
                    mode='lines',
                    text=ticker,
                    name=f"{ticker}_RSI"
                )

                # Assuming `x_values` is an array-like object that matches your existing data


                RSI_70_line = go.Scatter(x=x_values,
                                         y=[70]*len(x_values),
                                         mode='lines',
                                         name="70", 
                                         line = dict(dash='dash'))

                RSI_30_line = go.Scatter(x=x_values,
                                         y=[30]*len(x_values),
                                         mode='lines',
                                         name="30", 
                                         line = dict(dash='dash'))


                RSI_subplot_data.append(RSI_scatter)
                RSI_subplot_data.append(RSI_70_line)
                RSI_subplot_data.append(RSI_30_line)


# ARTICLES ****************************************************************************************
    
    if dropdown_value:
        for company in company_list:
            fetch_articles(company, article_dict, dt.datetime.strptime(start_date, '%Y-%m-%d').date(), dt.datetime.strptime(end_date, '%Y-%m-%d').date())
            article_df = article_dic_to_df(article_df, article_dict, company)
            article_df = article_abstract_sentiment(article_df, company)     
        company_to_ticker_index(article_df)
        
        for ticker in ticker_list:

            hover_texts = [
                    f"Headline: {headline} | Sentiment: {sentiment}"
                    for headline, sentiment in zip(
                        article_df[(f"{ticker}", "headline")].tolist(),
                        article_df[(f"{ticker}", "sentiment")].tolist())]
        
            article_dates = article_df[(f'{ticker}', 'pub_date')].tolist()
            y_values = []

            for pub_date in article_dates: 
                stock_price = data_dic[ticker]['Close'].get(pub_date, None)
                y_values.append(stock_price if stock_price is not None else 'NaN')


                article_scatter = go.Scatter(
                            x=article_dates,  # List of dates for articles
                            y=y_values,  # Replace with actual stock prices
                            mode='markers',
                            hoverinfo='text+x+y',
                            hovertext=hover_texts,
                            marker=dict(size=5, color='red'),
                            name='Articles'
                            )

            graph_data.append(article_scatter)
            
# Plot Figure ************************************************************************************

    # Calculate the number of rows needed for subplots
    num_rows = 1  
    if VMA_subplot_data:
        num_rows += 1
    if RSI_subplot_data:
        num_rows += 1

    # Initialize the subplot figure
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.009, horizontal_spacing=0.009)

    # Add traces for the main graph data
    for trace in graph_data:
        fig.add_trace(trace, row=1, col=1)

    # Dynamically add traces for VMA and RSI subplots
    current_row = 2  # Start from the second row
    if VMA_subplot_data:
        for trace in VMA_subplot_data:
            fig.add_trace(trace, row=current_row, col=1)
        current_row += 1  # Increment the current row for the next set of traces

    if RSI_subplot_data:
        for trace in RSI_subplot_data:
            fig.add_trace(trace, row=current_row, col=1)
        current_row += 1  # Increment the current row for the next set of traces

    # Update layout and return
    title = ', '.join(ticker_list)  # Update this as needed
    fig.update_layout(title=title, sliders=None)

    return fig
   


# In[10]:


if __name__ == '__main__':
    app.run_server(debug=False)


# Volume Moving Average (VMA):
# 
# What It Is: VMA is a simple moving average applied to trading volume rather than trading price. It helps to smooth out volume fluctuations and highlight underlying trends in trading activity.
#    
# How It's Used: VMA is often plotted as a line graph overlaid on a bar chart of trading volume or as a part of a standard price chart. It can help identify divergences between volume trends and price movements, providing insights into the strength of a trend.

# 2. Technical Indicators:
# 
# Implement popular technical indicators that traders use to make decisions.
# 
#     RSI (Relative Strength Index): Measures the speed and change of price movements.
#     MACD (Moving Average Convergence Divergence): Shows the relationship between two moving averages of a stock’s price.
#     Bollinger Bands: Uses standard deviations to determine if a stock is oversold or overbought.

# Certainly! The second part of the project, focusing on technical indicators, could indeed involve creating a set of graphs, each illustrating a different indicator. Here's a breakdown of what this could look like for AAPL or any other stock:
# 
# ### RSI (Relative Strength Index):
# - **What It Is:** RSI measures the speed and change of price movements and is usually plotted on a scale of 0 to 100.
# - **How to Display:** You could create a separate graph with a line representing the RSI over a specified time frame. Typically, levels above 70 are considered overbought, and levels below 30 are considered oversold. Highlighting these areas on the graph can help users identify potential buy or sell signals.
# 
# ### MACD (Moving Average Convergence Divergence):
# - **What It Is:** MACD shows the relationship between two moving averages of a stock’s price. It's represented by two lines: the MACD line (usually the 12-day EMA minus the 26-day EMA) and the Signal line (usually the 9-day EMA of the MACD line).
# - **How to Display:** This could also be a separate graph. Plot both the MACD and Signal lines, and optionally include a histogram showing the difference between them. Crossings of these lines can indicate potential trading signals.
# 
# ### Bollinger Bands:
# - **What It Is:** Bollinger Bands consist of a middle band (usually the 20-day SMA) with two outer bands that are standard deviations away from the middle.
# - **How to Display:** You could overlay Bollinger Bands on the existing Price over Time graph. When the price approaches or moves outside the bands, it might indicate overbought or oversold conditions.
# 
# ### Summary:
# You might create three separate graphs for RSI, MACD, and Bollinger Bands, or combine them in a single view, depending on the complexity and user experience you are aiming for. If you want to create an interactive dashboard, tools like Dash or Streamlit could allow users to toggle between different indicators or view them simultaneously.
# 
# Your approach should consider the audience's familiarity with these indicators. Including brief descriptions or interpretations of the graphs could make the analysis more accessible to users who are not expert traders.
# 
# Each of these indicators offers unique insights, and visualizing them through graphs can make those insights more intuitive and actionable for investors or analysts studying the stock.
# 

# In[ ]:


# Moving Average Convergence/Divergence (MACD)
AAPL_data['MACD'] = ta.EMA(AAPL_data['Close'], timeperiod=12) - ta.EMA(AAPL_data['Close'], timeperiod=26)
AAPL_data['MACD_Signal_Line'] = ta.EMA(AAPL_data['MACD'], timeperiod=9)
AAPL_data['MACD_Hist'] = AAPL_data['MACD'] - AAPL_data['MACD_Signal_Line']


# In[ ]:


#AAPL_data.index
AAPL_data["MACD_Hist"].max()


# In[ ]:


## Create subplots
ax1 = plt.subplot2grid((14,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((14,1), (5,0), rowspan = 4, colspan = 1, sharex = ax1)
ax3 = plt.subplot2grid((14,1), (9,0), rowspan = 4, colspan = 1, sharex = ax1)

## Price Over Time Graph
ax1.plot(AAPL_data['Close'], linewidth = 1.5)
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

## MACD Graph
ax2.plot(AAPL_data['MACD'], color="grey", linewidth = 1.0, label = 'MACD')
ax2.plot(AAPL_data['MACD_Signal_Line'], color="skyblue", linewidth = 1.0, label = 'Signal Line')
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

for i in range(len(AAPL_data['Close'])):
    
    if str(AAPL_data['MACD_Hist'][i])[0] == "-":
        ax2.bar(AAPL_data.index[i], AAPL_data['MACD_Hist'][i], color="tab:red")
    else: 
        ax2.bar(AAPL_data.index[i], AAPL_data['MACD_Hist'][i], color = "tab:green")

## Plot RSI subsplot
ax3.set_title('Relative Strength Indicator (RSI)', fontsize=10)
ax3.plot(AAPL_data['RSI'], color='tab:gray', linewidth = 1.0)
plt.axhline(y=70, color='tab:blue', linestyle="dashed", linewidth = 1.0)
plt.axhline(y=30, color='tab:orange', linestyle="dashed", linewidth = 1.0)
        
plt.xticks(rotation=45, ha='right')
plt.show()


# # Graveyard

# In[ ]:


ticker = ['AAPL']
stocks = yf.download(ticker, period = "2y")


# In[ ]:


stocks


# In[ ]:


#Price Over Time Graph
stocks['Close'].plot(figsize=(15,8), fontsize=13)
plt.title("AAPL Stock Price Over Time - 2 Year Period")
plt.ylabel("Price")
plt.legend(fontsize=13)
plt.show


# In[ ]:


# Moving Averages
APPL_Short_EMA_8 = ta.EMA(stocks['Close'], timeperiod = 8)
APPL_Short_EMA_20 = ta.EMA(stocks['Close'], timeperiod = 20)
APPL_Long_EMA_50 = ta.EMA(stocks['Close'], timeperiod = 50)
APPL_Long_EMA_200 = ta.EMA(stocks['Close'], timeperiod = 200)


# In[ ]:


stocks['Short_EMA_8'] = APPL_Short_EMA_8
stocks['Short_EMA_20'] = APPL_Short_EMA_20
stocks['Long_EMA_50'] = APPL_Long_EMA_50
stocks['Long_EMA_200'] = APPL_Long_EMA_200 
stocks


# In[ ]:


stocks[['Close', 'Short_EMA_8', 'Short_EMA_20']].plot(figsize=(15,8), fontsize=13)
plt.title("AAPL Stock Price Over Time - 2 Year Period")
plt.ylabel("Price")
plt.legend(fontsize=13)
plt.show

