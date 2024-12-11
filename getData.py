import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

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
symbol = "EURJPY"
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
df = data_frame['high']
# print(df)

df = np.array(df).reshape(-1,1)

df = scaler.fit_transform(df)
print(df)
# Shutdown MetaTrader 5
mt5.shutdown()
