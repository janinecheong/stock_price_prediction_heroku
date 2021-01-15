# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 10:42:00 2021

@author: sfgab / mjfc
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import set
#import seaborn as sns

# For calculation and web scrape
from math import ceil


# Import seed for reproducibility
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

# For preprocessing and model building
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

#date formatting
from datetime import datetime

#yahoo finance api
import yfinance as yf

# Import streamlit
#import streamlit as st
from streamlit import pyplot, subheader, write, title, markdown, sidebar, subheader


# Style settings
set()
plt.style.use('seaborn-white')




####################FUNCTIONS####################
def load_data():
    url = 'https://dailypik.com/top-50-companies-sp-500/'
    html = pd.read_html(url, header = 0) #pd.read_html is used since the data we want to scrape is already a table
    df = html[0] #specify that we only want the first table
    return df

def price_plot(symbol):
    df = pd.DataFrame(sp500[symbol]['Close'])
    df['Date'] = df.index
    f, ax = plt.subplots(figsize=(9, 7))
    plt.fill_between(df['Date'], df['Close'], color='skyblue', alpha=0.3)
    plt.plot(df['Date'], df['Close'], color='skyblue', alpha=1)
    plt.xticks(rotation=45)
    plt.title(symbol, fontweight='bold')
    plt.xlabel("Date", fontweight='bold')
    plt.ylabel("Closing Price", fontweight='bold')

#model = build_model(x_train, y_train)
def build_model(x_train, y_train):
    """parameters:
        x_train, y_train as processed training data after Data Preprocessing
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model


def model_prediction(model): ##### MODEL TESTING #####
    test_data = scaled_data[training_data_len - 60: , : ] #training_data_len minus 60 to the end of the dataset, and get all columns
    x_test = []
    y_test =  dataset[training_data_len : , : ] # y_test will be all of the values that we want our model to predict; actual test values; not scaled

    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0]) # Past 60 prices

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return [x_test, y_test, predictions]

def model_prediction_ja(model):
    test_data = scaled_data[training_data_len - 60: , : ] #training_data_len minus 60 to the end of the dataset, and get all columns

    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] # y_test will be all of the values that we want our model to predict; actual test values; not scaled

    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0]) # Past 60 prices
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return [x_test, y_test, predictions]


def data_viz(STOCK_PAR):
    #STOCK_PAR = STOCK_VAR - Name of the stock
    #constants:
        #train['Close']
        #actual_price[['Close', 'Predictions']]
    # Visualize the data
    train = data[:training_data_len]
    actual_price = data[training_data_len:]
    actual_price['Predictions'] = predictions
    fig = plt.figure(figsize=(16,8))
    plt.title(STOCK_PAR, fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Close Price USD ($)', fontsize=16)
    plt.plot(train['Close'])
    plt.plot(actual_price[['Close', 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right', fontsize=13)
    plt.show()
    pyplot(fig)
    subheader('Actual vs Predicted Closing Price')
    #will return the actual price vs prediction dataframe
    return actual_price

def get_metrics(stockname, actual, prediction):
    r2 = r2_score(actual, prediction)*100
    mae = mean_absolute_error(actual, prediction)
    rmse = np.sqrt(mean_squared_error(actual, prediction))

    print(stockname + ': ')
    print(("r2: %.2f") %r2)
    print(("mae: %.2f") %mae)
    print(("rmse: %.2f") %rmse)

    write(("R2: %.2f") %r2)
    write(("MAE: %.2f") %mae)
    write(("RMSE: %.2f") %rmse)


def pred_date(SYMBOL_PLACEHOLDER):
    date_today = datetime.today().strftime('%Y-%m-%d')

    # Get the quote
    # apple_quote = web.DataReader(SYMBOL_PLACEHOLDER, data_source='yahoo', start = '2015-01-01', end = date_today)
    apple_quote = pd.DataFrame(sp500[SYMBOL_PLACEHOLDER])

    # Create a new dataframe
    new_df = apple_quote.filter(['Close'])

    # Get the last 60 day closing price values and convert the dataframe to an array
    last_60_days = new_df[-60:].values

    # Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)

    # Create an empty list
    X_test = []

    # Append the past 60 days
    X_test.append(last_60_days_scaled)

    # Convert the X_test dataset to a numpy array
    X_test = np.array(X_test)

    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Get the predicted scaled price
    pred_price = model.predict(X_test)

    # Undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    write('Next trading day predicted closing price: $' + str(pred_price)[2:-2])

####################MAIN PROGRAM####################
# As of December 3, 2020, the following are the fifty largest S&P 500 index constituents by weight:

#stock_list = list(df['Symbol'])
#stock_list = ['DIS','AAPL']
#, 'PYPL', 'AMZN', 'ADBE','PFE'

#start_time = time.time()

date_today = datetime.today().strftime('%Y-%m-%d')

df = pd.read_csv('top_50_final.csv')
stock_list = list(df['Symbol'])
#date today
#date_today = datetime.today().strftime('%Y-%m-%d')
sp500 = yf.download(  # or pdr.get_data_yahoo(...
            # tickers list or string as well
            tickers = stock_list, interval = "1d", group_by = 'ticker', auto_adjust = True, prepost = True,
            threads = True, proxy = None,
            start = '2015-01-01',
            end = date_today
        )




###############STREAMLIT###########

title('Stock Price Prediction for the Top 50 S&P 500 Companies')

markdown("""
This app retrieves the list of the **Top 50 S&P 500** and its corresponding **actual closing price** and **predicted closing price** using LSTM.
* **Python libraries:** pandas, numpy, matplotlib, yfinance, sklearn, keras, streamlit
* **Data source:** Yahoo Finance, [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

sidebar.header('User Input Features')

streamlit_stock_selection = sidebar.selectbox(
    'Select stock symbol',
    stock_list)

if streamlit_stock_selection in stock_list:
    ####DATA PREPROCESSING####
    data = pd.DataFrame(sp500[streamlit_stock_selection]['Close'])
    dataset = data.values
    training_data_len = ceil( len(dataset) *.8) #use math.ceil to round up
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len , : ]
    x_train = [] #1143 1203-60, contains an array of 60 values per index
    y_train = [] #1143, contains single value only

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0]) #append past 60 days; contains 60 values from index 0 to 59
        y_train.append(train_data[i, 0]) #contains the 61st value which is at index 60

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    #################### MODEL BUILDING ####################
    model = build_model(x_train, y_train)

    #################### MODEL TESTING ####################

    x_test = model_prediction(model)[0]
    y_test = model_prediction(model)[1]
    predictions = model_prediction(model)[2]


    #################### DATA VISUALIZATION ####################
    subheader('Actual and Predicted Price of ' + streamlit_stock_selection + ' from 2015 to date')
    write(data_viz(streamlit_stock_selection))

    #################### PREDICTED DATA ####################
    pred_date(streamlit_stock_selection)

    ###METRICS###
    subheader("Model metrics")
    write(get_metrics(streamlit_stock_selection, y_test, predictions))


    #st.write(time.time() - start_time)
