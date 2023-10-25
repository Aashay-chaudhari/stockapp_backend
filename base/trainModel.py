import yfinance as yf

yf.pdr_override()
from pandas_datareader import data as pdr

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import math
# import numpy as np
# import pickle
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense, Dropout


df = pdr.get_data_yahoo("^NSEI", start="1980-02-01", end="2022-07-13")
print(df)
# column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
# df = df.reindex(columns=column_names)
# dataframe = df.reset_index()
# print(dataframe)
# data = df.filter(['Close'])
# dataset = data.values
# # type(dataset), type(data)
# training_data_len = math.ceil(len(dataset) * .8)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)
# train_data = scaled_data[0:training_data_len, :]
# x_train = []
# y_train = []
# for i in range(60, len(train_data)):
#     x_train.append(train_data[i - 60:i, 0])
#     y_train.append(train_data[i, 0])
# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape, y_train.shape)
#
# model1 = Sequential()
# model1.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model1.add(LSTM(units=50, return_sequences=False))
# model1.add(Dense(units=25))
# model1.add(Dense(units=1))
#
# model1.compile(optimizer='adam', loss='mean_squared_error')
# ## epochs = 9
# model1.fit(x_train, y_train, batch_size=64, epochs=100)
# keras_model_path = '/api/keras_save'
# model1.save(keras_model_path)
