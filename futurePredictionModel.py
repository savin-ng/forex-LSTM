import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import MetaTrader5 as mt5
from datetime import datetime

# Function to initialize MetaTrader 5
def initialize_mt5():
    path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    login = 88723119
    password = "BuCvF_B7"
    server = "MetaQuotes-Demo"
    timeout = 10000
    portable = True
    mt5.initialize(path=path, login=login, password=password, server=server, timeout=timeout, portable=portable)
    return mt5

mt5 = initialize_mt5()

# Define the symbol and timeframe
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M30

# Function to retrieve data
def get_data(start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    return rates_frame

# Define the date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# Fetch the data and shut down MetaTrader 5
data_frame = get_data(start_date, end_date)
mt5.shutdown()

# Select the 'high' column for prediction
data = np.array(data_frame['high']).reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split into training and test sets
split = int(len(data) * 0.8)
train, test = data[:split], data[split:]

# Function to create input-output sequences
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Set look-back window
look_back = 30

# Prepare training and test data
x_train, y_train = create_sequences(train, look_back)
x_test, y_test = create_sequences(test, look_back)

# Reshape inputs for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1)

# Evaluate the model
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Visualize the predictions
plt.figure(figsize=(15, 7))
plt.title('Forex Rate Prediction')
plt.plot(y_test, label='Actual', color='g')
plt.plot(y_pred, label='Predicted', color='r')
plt.legend()
plt.show()

# Predict 1000 timesteps into the future
future_steps = 1000
last_sequence = data[-look_back:]
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, look_back, 1))
    future_predictions.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
print(future_predictions)

# Plot future predictions
plt.figure(figsize=(15, 7))
plt.title('Future Forex Rate Prediction')
plt.plot(range(len(data)), scaler.inverse_transform(data), label='Historical', color='b')
plt.plot(range(len(data), len(data) + future_steps), future_predictions, label='Future', color='orange')
plt.legend()
plt.show()
