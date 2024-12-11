import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import mean_squared_error
import MetaTrader5 as mt5
from datetime import datetime

def initialize_mt5():
    # Initialize connection to MetaTrader 5
    path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    login = 88723119
    password = "BuCvF_B7"
    server = "MetaQuotes-Demo"
    timeout = 10000
    portable = True
    mt5.initialize(path=path, login=login, password=password, server=server, timeout=timeout, portable=portable)
    return mt5 

mt5 = initialize_mt5()

# Define the symbol
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M30  # 1 hour timeframe for backtesting


# Function to simulate trading
def get_Data(start_date, end_date):
    # Get historical data
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    rates_frame = pd.DataFrame(rates)
    print(rates_frame.shape)
    print(rates_frame.columns)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    print(rates_frame.head(100))
    return rates_frame

# Define the date range for backtesting
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

data_frame = get_Data(start_date, end_date)
# Shutdown MetaTrader 5
mt5.shutdown()

scaler = MinMaxScaler()
df = data_frame['high']
df = np.array(df).reshape(-1,1)
df = scaler.fit_transform(df)

#Training and test sets
train = df[:10000]
test = df[10000:]

# print(train.shape)
# print(test.shape)

def get_data(data, look_back):
  datax, datay = [],[]
  for i in range(len(data)-look_back-1):
    datax.append(data[i:(i+look_back),0])
    datay.append(data[i+look_back,0])
  return np.array(datax) , np.array(datay)

look_back = 1

x_train , y_train = get_data(train, look_back)
# print(x_train.shape)
# print(y_train.shape)

x_test , y_test = get_data(test,look_back)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

# print(x_train.shape)
# print(x_test.shape)

n_features=x_train.shape[1]
model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(1,1)))
model.add(Dense(n_features))

model.compile(optimizer='adam', loss = 'mse')

model.fit(x_train,y_train, epochs = 100, batch_size=64)

scaler.scale_

y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

y_test = np.array(y_test).reshape(-1,1)
y_test = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

plt.figure(figsize=(20,8))
plt.title('Foreign Exchange Rate of Euro')
plt.plot(y_test , label = 'Actual', color = 'g')
plt.plot(y_pred , label = 'Predicted', color = 'r')
plt.legend()
plt.show()