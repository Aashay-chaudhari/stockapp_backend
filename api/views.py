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
model = tf.keras.models.load_model('my_new_model')
print("Model is loaded in global")


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


# Predict next day closing price based on last 60 days
@api_view(['POST'])
def predict(request):
    if request.data["us_stock"] == True:
        stockName = request.data["symbol"]
    else:
        stockName = request.data["symbol"] + ".NS"
    print("stock name is: ", stockName)
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

    pred = model.predict(scaled_df)

    pred_transform = inverse_scale_target(pred, min_var, max_var)
    print("pred_transform is: ", pred_transform)
    print("last closing price: ", df["Close"][-1])
    return Response({
        'predicted_price': pred_transform,
        'last_closing_price' : df["Close"][-1]
    })

@api_view(['POST'])
def show_similar(request):
    if request.data["us_stock"] == True:
        stockName = request.data["symbol"]
    else:
        stockName = request.data["symbol"] + ".NS"
    df = pdr.get_data_yahoo(stockName, start="2010-02-01", end=current_time.strftime('%Y-%m-%d'))
    df.reset_index(inplace=True)
    df['target'] = np.where(df['Open'].shift(-1) > df['Close'], 1, 0)
    print(df.head(5))
    target = df['target']
    epsilon = 0.000000001
    feature = df['Open']
    samples_to_return = []
    samples = []
    sample_targets = []
    for i in range(0, len(feature) - 15):
        sample = feature[i:i + 15]
        samples_to_return.append(df[i:i+15])
        samples.append(sample)
        sample_targets.append(target.iloc[i + 14])

    samples_array = np.array(samples)
    print("sample array example: ", samples_array[0])
    print("samples_to_return array example: ", samples_to_return[0])

    scaled_samples = []

    for sample in samples:
        max_in_sample = max(sample)
        min_in_sample = min(sample)
        buffer_sample = []
        for data in sample:
            buffer_data = (data - min_in_sample) / (max_in_sample - min_in_sample + epsilon)
            buffer_sample.append(buffer_data)
        scaled_samples.append(buffer_sample)
    print("scaled sample array examples:", scaled_samples[0])
    print("count of scaled samples and targets", np.array(scaled_samples).shape, len(sample_targets))
    sse = {}

    target_seq = scaled_samples[-1]

    for i in range(0, len(scaled_samples)):
        sample = scaled_samples[i]
        errors = np.subtract(sample, target_seq)
        sse_val = 0
        for j in range(0, len(errors)):
            error = errors[j]
            sse_val += j * 0.1 * error * error
        sse[i] = sse_val

    sorted_dict = dict(sorted(sse.items(), key=lambda item: item[1]))

    counter = 0
    targets = []
    return_keys = []
    for key in sorted_dict:
        if counter < 4:
            return_keys.append(key)
            targets.append(sample_targets[key])
        else:
            break
        counter += 1

    print(targets)
    print("Keys to return: ", return_keys)
    targets_seq = pd.DataFrame(targets[1:])
    counts = targets_seq.value_counts()

    if counts[1] > counts[0]:
        print("1 is the label")
    else:
        print("0 is the label")

    return_targets = []

    for sig in targets:
        if sig == 1:
            return_targets.append("BUY")
        else:
            return_targets.append("SELL")
    print("Function execution is over.")
    return Response({
        'chart0': samples_to_return[return_keys[0]],
        'chart1': samples_to_return[return_keys[1]],
        'chart2': samples_to_return[return_keys[2]],
        'chart3': samples_to_return[return_keys[3]],
        'signals': return_targets
    })



# Get data for individual stock symbols
@api_view(['POST'])
def getStockData(request):
    stockName = request.data["symbol"] + ".NS"
    df = pdr.get_data_yahoo(stockName, start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
    df50_with_dates = df.reset_index()
    df_data = df50_with_dates
    return Response(df_data)


# For signals page, to understand model prediction on given stock
@api_view(['POST'])
def getModelData(request):
    stockName = request.data["symbol"] + ".NS"
    df = pdr.get_data_yahoo(stockName, start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
    print("df columns before switch: ", df.columns)
    column_names = ["Open", "Close"]
    df = df.reindex(columns=column_names)
    dataframe = np.array(df[-170:])
    print("dataframe is: ", dataframe, dataframe.shape)
    batches_testing = []
    targets_testing = []

    for i in range(30, dataframe.shape[0]):
        buffer_array = list(dataframe[i - 30:i])
        targets_testing.append(dataframe[i][0])
        batches_testing.append(buffer_array)

    print("shape of testing and target training: ", np.array(batches_testing).shape, np.array(targets_testing).shape)

    # Define batch scaling function
    scaled_testing_data = []
    scaled_testing_targets = []

    scaling_dict = {}

    def inverse_scale_data(x, min_x, max_x):
        orig_seq = []
        for j in x:
            buffer_array = []
            for k in j:
                orig_k = k * (max_x - min_x) + min_x
                buffer_array.append(orig_k)
            orig_seq.append(buffer_array)
        return orig_seq

    def inverse_scale_target(x, min_x, max_x):
        orig_x = x * (max_x - min_x) + min_x
        return orig_x

    def scale(x, buffer_target, i):
        seq_x = np.array(x)
        max_x = np.amax(seq_x)
        min_x = np.amin(seq_x)
        scaling_dict[i] = (min_x, max_x)
        new_seq = []
        for j in seq_x:
            buffer_seq = []
            for k in j:
                new_k = (k - min_x) / (max_x - min_x)
                buffer_seq.append(new_k)
            new_seq.append(buffer_seq)
        scaled_target = (buffer_target - min_x) / (max_x - min_x)
        return new_seq, scaled_target

    for i in range(0, len(batches_testing)):
        seq = batches_testing[i]
        buffer_target = targets_testing[i]
        new_seq, scaled_target = scale(seq, buffer_target, i)
        scaled_testing_data.append(new_seq)
        scaled_testing_targets.append(scaled_target)

    print("shape of scaled testing and target", np.array(scaled_testing_data).shape,
          np.array(scaled_testing_targets).shape)

    print("Scaling dict len: ", len(scaling_dict.keys()))

    x_test = np.array(scaled_testing_data)
    y_test = np.array(scaled_testing_targets)

    print(x_test.shape, y_test.shape)

    prediction_list = model.predict(x_test)

    transformed_prediction_list = []

    for i in range(0, len(prediction_list)):
        dict_values = list(scaling_dict[i])
        or_seq = inverse_scale_target(prediction_list[i], dict_values[0], dict_values[1])
        transformed_prediction_list.append(or_seq)

    print("prediction shape", len(transformed_prediction_list), np.array(df["Open"][-140:]).shape)
    #

    prediction_list = transformed_prediction_list
    actual_values = np.array(df["Open"][-140:])
    print("shape of actual_values: ", actual_values.shape)

    actual_values_dict = dict()
    for index, value in enumerate(actual_values):
        actual_values_dict[index] = value
    predicted_values_dict = dict()
    for index, value in enumerate(prediction_list):
        predicted_values_dict[index] = value
    ResponseDataframe = pd.DataFrame({'actual': actual_values_dict, 'predicted': predicted_values_dict})
    ResponseDataframe = ResponseDataframe.fillna('')
    predicted_price = [10000]
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
