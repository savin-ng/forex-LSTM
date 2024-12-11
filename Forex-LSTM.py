import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import mean_squared_error

data_set = pd.read_csv(r'E:\University\Semester_7\FYP\Model\LSTM\Foreign_Exchange_Rates.csv', na_values='ND')
print(data_set.shape)
print(data_set.head(10))

data_set.isnull().sum()
data_set.interpolate(inplace=True)
data_set.isnull().sum()

# plt.plot(data_set['EURO AREA - EURO/US$'])
# plt.show()

df = data_set['EURO AREA - EURO/US$']
# print(df)

df = np.array(df).reshape(-1,1)

df = scaler.fit_transform(df)

#Training and test sets
train = df[:4800]
test = df[4800:]

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

model.fit(x_train,y_train, epochs = 100, batch_size=16)

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