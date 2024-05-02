#Import Utility
import numpy as np
import math
from pandas_datareader import data as pdr

#Import helper files
from base.helperClasses.new_preprocessor import Preprocessor
import base.params as params

#Import Machine Learning Packages
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf


def train_model(stock_name : str = '', model_name : str = "test_model") -> str:
    preprocessor = Preprocessor()
    buffer_df = preprocessor.get_data_and_create_features()
    x_train, y_train, x_test, y_test = preprocessor.create_scaled_and_balanced_batches(df = buffer_df)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    ## epochs = 9
    history = model.fit(x_train, y_train, batch_size=256, epochs=1, validation_split=0.2)

    print("Saving model")
    keras_model_path = params.BASE_DIR_PATH / "base/model_store/models" / model_name

    model.save(keras_model_path)

    print("Model saved")

    response = "model has been successfully trained and saved at " + keras_model_path
    return response

