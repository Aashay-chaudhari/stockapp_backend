# Import libraries

# Import utility libraries
from datetime import datetime
import math
import matplotlib.pyplot as plt

# Import django related libraries
from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Stock, UserModel
from .serializers import StockSerializer, UserModelSerializer

# Import Machine Learning Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras import Model
from keras.layers import Input, Dense, Dropout, Layer, LSTM
import tensorflow as tf
import keras
import keras.backend as K

# Import data import libraries
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

# Global variables to be used in this file
current_time = datetime.now()


def trainModel_with_new_scaling(request):
    df = pdr.get_data_yahoo("^NSEI", start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
    feature = np.array(df[['Open', 'Close']])

    print(feature.shape)

    training_data_len = math.ceil(feature.shape[0] * 0.8)
    print(training_data_len)
    training_data = feature[:training_data_len]
    testing_data = feature[training_data_len:]
    print(training_data.shape, testing_data.shape)
    batches_training = []
    targets_training = []

    for i in range(30, len(training_data)):
        buffer_array = list(training_data[i - 30:i])
        targets_training.append(training_data[i][0])
        batches_training.append(buffer_array)

    print(np.array(batches_training).shape, np.array(targets_training).shape)

    # Define batch scaling function
    scaled_training_data = []
    scaled_training_targets = []

    def scale(x, buffer_target):
        seq_x = np.array(x)
        max_x = np.amax(seq_x)
        min_x = np.amin(seq_x)
        new_seq = []
        for j in seq_x:
            buffer_seq = []
            for k in j:
                new_k = (k - min_x) / (max_x - min_x)
                buffer_seq.append(new_k)
            new_seq.append(buffer_seq)
        scaled_target = (buffer_target - min_x) / (max_x - min_x)
        return new_seq, scaled_target

    for i in range(0, len(batches_training)):
        seq = batches_training[i]
        buffer_target = targets_training[i]
        new_seq, scaled_target = scale(seq, buffer_target)
        scaled_training_data.append(new_seq)
        scaled_training_targets.append(scaled_target)

    x_train = np.array(scaled_training_data)
    y_train = np.array(scaled_training_targets)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    ## epochs = 9
    history = model.fit(x_train, y_train, batch_size=256, epochs=40, validation_split=0.2)

    print("Saving model")
    keras_model_path = 'my_new_model'
    model.save(keras_model_path)
    print("Model saved")


# #Trains the model on data, and saves the model in my_model
# def trainModel():
#     df = pdr.get_data_yahoo("^NSEI", start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
#     column_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
#     df = df.reindex(columns=column_names)
#     dataframe = df.reset_index()
#     data = df.filter(['Close'])
#     dataset = data.values
#     training_data_len = math.ceil(len(dataset) * .8)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(dataset)
#     train_data = scaled_data[0:training_data_len, :]
#     #Create training data
#     x_train = []
#     y_train = []
#     for i in range(60, len(train_data)):
#         x_train.append(train_data[i - 60:i, 0])
#         y_train.append(train_data[i, 0])
#     x_train, y_train = np.array(x_train), np.array(y_train)
#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#     print(x_train.shape, y_train.shape)
#
#     #Create testing data
#     x_test = []
#     y_test = []
#     for i in range(60, len(train_data)):
#         x_test.append(train_data[i - 60:i, 0])
#         y_test.append(train_data[i, 0])
#     x_test, y_test = np.array(x_test), np.array(y_test)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#     print(x_test.shape, y_test.shape)
#     # Describe model architecture
#
#     class attention(Layer):
#         def __init__(self, **kwargs):
#             super(attention, self).__init__(**kwargs)
#
#         def build(self, input_shape):
#             self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
#                                      initializer='random_normal', trainable=True)
#             self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
#                                      initializer='zeros', trainable=True)
#             super(attention, self).build(input_shape)
#
#         def call(self, x):
#             # Alignment scores. Pass them through tanh function
#             e = K.tanh(K.dot(x, self.W) + self.b)
#             # Remove dimension of size 1
#             e = K.squeeze(e, axis=-1)
#             # Compute the weights
#             alpha = K.softmax(e)
#             # Reshape to tensorFlow format
#             alpha = K.expand_dims(alpha, axis=-1)
#             # Compute the context vector
#             context = x * alpha
#             context = K.sum(context, axis=1)
#             return context
#
#     def create_LSTM_with_attention(hidden_units, dense_units):
#         x = Input(shape=(x_train.shape[1:]))
#         conv_x = keras.layers.Conv1D(30, 1, activation='relu')(x)
#         attention_layer = attention()(conv_x)
#         print(attention_layer.shape, attention_layer)
#         dropout_lstm = keras.layers.Dropout(.2)(attention_layer)
#         reshaped_attention = keras.layers.Reshape((30, 1), input_shape=(30,))(dropout_lstm)
#         batchnorm_reshaped_attention = keras.layers.BatchNormalization()(reshaped_attention)
#         lstm_layer = LSTM(100, return_sequences=True, activation='relu')(batchnorm_reshaped_attention)
#         lstm_layer = LSTM(50, return_sequences=False, activation='relu')(lstm_layer)
#         outputs = Dense(1, trainable=True, activation='tanh')(lstm_layer)
#         model = Model(x, outputs)
#         model.compile(loss='mse', optimizer='adam')
#         return model
#
#         # Create the model with attention, train and evaluate
#
#     model_attention = create_LSTM_with_attention(hidden_units=100, dense_units=1)
#
#     model_attention.summary()
#
#     model_attention.fit(x_train, y_train, epochs=25, batch_size=32, verbose=2, validation_split=0.2)
#
#     preds = model_attention.predict(x_test)
#
#     plt.plot(y_test)
#     plt.plot(preds)
#     plt.savefig('myfig.png')
#
#     print("Saving model")
#     keras_model_path = 'my_model'
#     model_attention.save(keras_model_path)
#     print("Model saved")

# Predict next day closing price based on last 60 days
@api_view(['POST'])
def predict(request):
    stockName = request.data["symbol"] + ".NS"
    df = yf.download(stockName, '2022-01-01', end=current_time.strftime('%Y-%m-%d'))
    column_names = ["Open", "Close"]
    df = df.reindex(columns=column_names)
    df = df[-30:]
    print("df is: ", df)

    def inverse_scale_target(x, min_x, max_x):
        orig_x = x * (max_x - min_x) + min_x
        return orig_x

    def scale(x, i):
        seq_x = np.array(x)
        max_x = np.amax(seq_x)
        min_x = np.amin(seq_x)
        new_seq = []
        for j in seq_x:
            buffer_seq = []
            for k in j:
                new_k = (k - min_x) / (max_x - min_x)
                buffer_seq.append(new_k)
            new_seq.append(buffer_seq)
        return new_seq, min_x, max_x

    scaled_df, min_var, max_var = scale(df, 0)
    scaled_df = np.array(scaled_df)
    print("scaled df shape is : ", scaled_df.shape)
    scaled_df = np.reshape(scaled_df, (1, scaled_df.shape[0], scaled_df.shape[1]))
    print("scaled df post reshape shape is : ", scaled_df.shape)

    model = tf.keras.models.load_model('my_new_model')
    pred = model.predict(scaled_df)

    print("pred is: ", pred)

    pred_transform = inverse_scale_target(pred, min_var, max_var)

    return Response({
        'predicted_price': pred_transform
    })


# Get data for individual stock symbols
@api_view(['POST'])
def getStockData(request):
    stockName = request.data["symbol"] + ".NS"
    df = pdr.get_data_yahoo(stockName, start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
    df50_with_dates = df.reset_index()
    df_data = df50_with_dates
    return Response(df_data)

#For signals page, to understand model prediction on given stock
@api_view(['POST'])
def getModelData(request):
    stockName = request.data["symbol"] + ".NS"
    df = pdr.get_data_yahoo(stockName, start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
    print("df columns before switch: ", df.columns)
    column_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    dataframe = df.reset_index()
    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    model = tf.keras.models.load_model('my_model')

    df_for_testing = dataframe[-200:]
    df_for_testing_scaled = scaled_data[-200:]
    xtest = []
    ytest = []

    for i in range(60, len(df_for_testing)):
        xtest.append(df_for_testing_scaled[i - 60:i, 0])
        ytest.append(df_for_testing_scaled[i, 0])
    xtest, ytest = np.array(xtest), np.array(ytest)
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
    print("xtest shape: ", xtest.shape)
    prediction_list = []

    for i in xtest:
        i = np.reshape(i, (1, 60, 1))
        price = model.predict(i)
        prediction_list.append(price)

    predicted_price_dataset = np.array(df_for_testing_scaled[-60:]).reshape(1, 60, 1)
    predicted_price = model.predict(predicted_price_dataset)
    predicted_price = scaler.inverse_transform(predicted_price)
    prediction_list = np.array(prediction_list)
    prediction_list = np.reshape(prediction_list, (prediction_list.shape[0], prediction_list.shape[1]))
    prediction_list = scaler.inverse_transform(prediction_list)
    test_values = df_for_testing['Close']
    actual_values = test_values[60:]
    actual_values = np.array(actual_values)
    actual_values_dict = dict()
    for index, value in enumerate(actual_values):
        actual_values_dict[index] = value
    predicted_values_dict = dict()
    for index, value in enumerate(prediction_list):
        predicted_values_dict[index] = value
    ResponseDataframe = pd.DataFrame({'actual': actual_values_dict, 'predicted': predicted_values_dict})
    ResponseDataframe = ResponseDataframe.fillna('')
    print("returning response", ResponseDataframe, predicted_price)
    return Response({
        'response': ResponseDataframe,
        'predicted_price': predicted_price
    })


@api_view(['GET'])
def getData(request):
    stocks = Stock.objects.all()
    serializer = StockSerializer(stocks, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def getUserData(request):
    users = UserModel.objects.all()
    serializer = UserModelSerializer(users, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def addStock(request):
    print("Data is: ", request)
    print("Data from req is: ", request.data)
    print("Name is: ", request.data["name"])
    serializer = StockSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)


@api_view(['POST'])
def addUser(request):
    print("Inside add user")
    serializer = UserModelSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)


@api_view(['POST'])
def checkLogin(request):
    loginStatus = False
    users = UserModel.objects.all()
    serializer = UserModelSerializer(users, many=True)
    username = request.data["name"]
    pass1 = request.data["password"]
    for user in users:
        print(user.name, user.password)
        if (user.name == username):
            if (user.password == pass1):
                loginStatus = True
                return Response("Success")

    return Response("Failed")
