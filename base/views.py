# Import utility libraries
from datetime import datetime
import math
import os
from django.conf import settings
import base.params as params
import numpy as np
import pandas as pd

# Import django related libraries
from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Stock, UserModel
from base.serializers import StockSerializer, UserModelSerializer

# Import Machine Learning Libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

# Import data import libraries
from pandas_datareader import data as pdr
import yfinance as yf


# Import classes
from base.helperClasses.user_auth import UserAuth
from base.helperClasses.helper_func import HelperFunc
from base.functions.train_model import train_model

# Global variables to be used in this file
user_auth = UserAuth()

#Load model
current_time = datetime.now()
yf.pdr_override()

# model = tf.keras.models.load_model('my_new_model')
model_path = os.path.join(params.BASE_DIR_PATH, 'base/model_store/models/model_001.h5')
new_model = tf.keras.models.load_model(model_path)
new_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

print("Model is loaded in global")

check_user_auth = True
seq_len = 30


def trainModel_with_new_scaling(request):
    response = train_model()
    return Response({
        'predicted_price': response
    })


# Predict next day closing price based on last 60 days
@api_view(['POST'])
def predict(request):
    user_name = request.data["user_name"]
    access_token = request.data["access_key"]

    if not UserAuth.check_if_user_active(user_name, access_token):
        return Response("login required")

    stock_name = HelperFunc.check_if_nse_stock(request)

    print("Calling Helper function predict")
    prediction, last_close = HelperFunc.predict(stock_name, new_model)

    return Response({
        'predicted_price': prediction,
        'last_closing_price': last_close
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
        samples_to_return.append(df[i:i + 15])
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
    print("Targets seq is: ", targets_seq)
    counts = targets_seq.value_counts()
    print("counts is : ", counts, type(counts), len(counts))

    counts_list = dict(counts)

    print("counts_list is: ", counts_list)
    if (0,) not in counts_list:
        counts_list[(0,)] = 0

    if (1,) not in counts_list:
        counts_list[(1,)] = 0

    if counts_list[(1,)] > counts_list[(0,)]:
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


# For signals page, to understand model prediction on given stock
@api_view(['POST'])
def get_model_data(request):
    stock_name = HelperFunc.check_if_nse_stock(request)

    predictions, actual_values, date_array = HelperFunc.get_model_results_for_stock(stock_name, new_model)
    print("predictions, actual values", len(predictions), len(actual_values))
    actual_values_dict = dict()
    for index, value in enumerate(actual_values):
        actual_values_dict[index] = value
    predicted_values_dict = dict()
    for index, value in enumerate(predictions):
        predicted_values_dict[index] = value
    dates_values_dict = dict()
    for index, value in enumerate(date_array):
        dates_values_dict[index] = value
    response_dataframe = pd.DataFrame({'actual': actual_values_dict, 'predicted': predicted_values_dict, 'dates': dates_values_dict})

    return Response({
        'response': response_dataframe
    })


# Get data for individual stock symbols
@api_view(['POST'])
def get_stock_data(request):
    stock_name = request.data["symbol"] + ".NS"
    df = pdr.get_data_yahoo(stock_name, start="1980-02-01", end=current_time.strftime('%Y-%m-%d'))
    df50_with_dates = df.reset_index()
    df_data = df50_with_dates
    return Response(df_data)


@api_view(['GET'])
def get_data(request):
    stocks = Stock.objects.all()
    serializer = StockSerializer(stocks, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def get_user_data(request):
    users = UserModel.objects.all()
    serializer = UserModelSerializer(users, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def add_stock(request):
    serializer = StockSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    return Response(serializer.data)


@api_view(['POST'])
def add_user(request):
    serializer = UserModelSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    else:
        print(serializer.errors)
        return Response("Email")
    return Response(serializer.data)


@api_view(['POST'])
def check_login(request):
    return Response(UserAuth.check_creds(request))
