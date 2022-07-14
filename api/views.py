from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Stock, UserModel
from .serializers import StockSerializer, UserModelSerializer
import yfinance as yf  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import tensorflow as tf
import pandas_datareader as web

from sklearn.preprocessing import StandardScaler

def trainModel():
    df = web.DataReader('^NSEI', data_source='yahoo', start='2000-01-3', end='2022-05-25')
    column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    dataframe = df.reset_index()
    print(dataframe)
    data = df.filter(['Close'])
    dataset = data.values
    # type(dataset), type(data)
    training_data_len = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape, y_train.shape)
    
    model1 = Sequential()
    model1.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model1.add(LSTM(units=50, return_sequences=False))
    model1.add(Dense(units=25))
    model1.add(Dense(units=1))

    model1.compile(optimizer='adam', loss='mean_squared_error')
    ## epochs = 9
    model1.fit(x_train, y_train, batch_size=64, epochs=100)
    keras_model_path = '/api/keras_save'
    model1.save(keras_model_path)

@api_view(['POST'])
def getModelData(request):
    print('hi')
    print("request data: ", request.data)
    stockName = request.data["symbol"] + ".NS"
    df = yf.download(stockName,'2005-01-01','2021-10-10')
    column_names = ["Close", "High", "Low", "Open", "Adj Close", "Volume"]
    df = df.reindex(columns=column_names)
    dataframe = df.reset_index()
    print(dataframe)
    data = df.filter(['Close'])
    dataset = data.values
    # type(dataset), type(data)
    training_data_len = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    keras_model_path = '/api/keras_save'
    model1 = tf.keras.models.load_model(keras_model_path)

    df_for_testing = dataframe[-200:]
    df_for_testing_scaled = scaled_data[-200:]
    xtest = []
    ytest = []

    for i in range(60, len(df_for_testing)):
        xtest.append(df_for_testing_scaled[i - 60:i, 0])
        ytest.append(df_for_testing_scaled[i, 0])
    xtest, ytest = np.array(xtest), np.array(ytest)
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
    print("reshape done")
    prediction_list = []

    for i in xtest:
        print("Inside testing loop")
        print(i)
        i = np.reshape(i, (1, 60, 1))
        print("i after reshape: ", i)
        price = model1.predict(i)
        prediction_list.append(price)
    print("Before predicting tomorrow value")
    print(xtest)
    print(xtest[1])
    predicted_price_dataset = df_for_testing_scaled[-60:]
    predicted_price_dataset = np.array(predicted_price_dataset)
    predicted_price_dataset = np.reshape(predicted_price_dataset, (predicted_price_dataset.shape[0], predicted_price_dataset.shape[1], 1))
    predicted_price_dataset = np.reshape(predicted_price_dataset, (1,60,1))
    predicted_price = model1.predict(predicted_price_dataset)

    predicted_price = scaler.inverse_transform(predicted_price)
    prediction_list = np.array(prediction_list)
    prediction_list = np.reshape(prediction_list, (prediction_list.shape[0], prediction_list.shape[1]))
    prediction_list = scaler.inverse_transform(prediction_list)
    test_values = df_for_testing['Close']
    actual_values = test_values[60:]
    actual_values = np.array(actual_values)
    actual_values_dict = dict() 
    for index,value in enumerate(actual_values):
        actual_values_dict[index] = value
    print(actual_values_dict)
    predicted_values_dict = dict() 
    for index,value in enumerate(prediction_list):
        predicted_values_dict[index] = value
    print(predicted_values_dict)
    ResponseDataframe = pd.DataFrame({'actual': actual_values_dict, 'predicted': predicted_values_dict})
    print("Finished creating data frame: ", ResponseDataframe)
    ResponseDataframe = ResponseDataframe.fillna('')
    return Response({
        'response' : ResponseDataframe,
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
    serializer = StockSerializer(data = request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)

@api_view(['POST'])
def addUser(request):
    print("Inside add user")
    serializer = UserModelSerializer(data = request.data)
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
        if(user.name == username):
            if(user.password == pass1):
                loginStatus = True
                return Response("Success")           
        
    return Response("Failed")

@api_view(['POST'])
def getStockData(request):
    print("Stock symbol is: ", request.data["symbol"])
    stockName = request.data["symbol"] + ".NS"
    Nifty50 = yf.download(stockName,'2010-01-01','2022-06-06')
    print(Nifty50.tail())
    Nifty50_with_dates = Nifty50.reset_index()
    Nifty50_data = Nifty50_with_dates

    return Response(Nifty50_data)
    