# Import Data Import libraries
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

yf.pdr_override()
import pandas_ta as pta

# Utility functions
import numpy as np
import math
import random

# Import plotting functions
import matplotlib.pyplot as plt


class DataImport:
    def __init__(self, stock_name='IBM'):
        self.stock_name = stock_name
        self.original_df = None

    def get_data(self, flag='default', seq_len=30):
        self.original_df = pdr.get_data_yahoo(self.stock_name, start="1980-02-01",
                                              end=datetime.now().strftime('%Y-%m-%d'))
        print("tail is: ", self.original_df.tail())
        if flag == 'predict':
            return self.original_df[-seq_len:]
        if flag == 'get_model_results':
            return self.original_df[-150:]
        return self.original_df

    def create_features(self, df):
        print("creating features for df: ", df.shape, type(df))
        df['diff'] = (df['Close'] - df['Open']) * 100 / df['Open']
        df['EWMA10'] = df['Close'].ewm(span=10).mean()
        df['EWMA5'] = df['Close'].ewm(span=5).mean()
        sti = pta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3)
        df['supertrend'] = sti['SUPERTd_7_3.0']
        print("features created")
        return df


class TransformData:
    def __init__(self, seq_len=30, cv=2):
        self.master_df = None
        self.seq_len = seq_len
        self.batches = None
        self.targets = None
        self.scaling_dict = {}
        self.cv = cv
        self.testing_targets = None  ##Testing targets batches to get last close price
        self.unscaled_targets = None
        self.testing_dates = None

    def transformation_workflow(self, df, show_sample_scale_example=True, predict=False):
        self.master_df = df
        self.create_batches()
        x_train, y_train, x_test, y_test = self.custom_scaling(*self.create_train_test())
        if show_sample_scale_example:
            print("Inside plotting function")
            self.show_sample_plot(x_test)
        return x_train, y_train, x_test, y_test

    def show_sample_plot(self, x_test):
        index = random.randint(0, x_test.shape[0])
        min_series, max_series = self.scaling_dict[index]
        random_series = x_test[index]
        transformed_series = []
        random_series_open = []
        for val in random_series:
            print("val is: ", val)
            print("val[0] is: ", val[0])
            random_series_open.append(val[0])
            transformed_series.append(val[0] * (max_series - min_series) + min_series)

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

        # Plot data on the respective subplots
        axs[0].plot(random_series_open)
        axs[0].set_title('Scaled Series')

        axs[1].plot(transformed_series)
        axs[1].set_title('Original Series')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plots
        plt.show()

    def create_batches(self, df=None):
        if df is None:
            buffer_df = np.array(self.master_df)
        else:
            buffer_df = np.array(df)

        buffer_array_batch = []
        buffer_array_target = []
        for i in range(self.seq_len, len(buffer_df)):
            buffer_array = buffer_df[i - self.seq_len:i, :]
            buffer_array_batch.append(buffer_array)
            buffer_array_target.append(buffer_df[i][1])

        self.batches, self.targets = np.array(buffer_array_batch), np.array(buffer_array_target)

        self.target_counts(self.batches, self.targets, 'total dataset')

        print("Number of batches and targets : ", self.batches.shape, self.targets.shape)
        return self.batches, self.targets

    def create_train_test(self, split_ratio=0.8, random_flag=True):
        if random_flag:
            training_data_len = math.ceil(self.batches.shape[0] * split_ratio)
            training_indexes = random.sample(range(0, self.batches.shape[0]), training_data_len)

            x_train, y_train, x_test, y_test = [], [], [], []

            for index, series in enumerate(self.batches):
                if index in training_indexes:
                    x_train.append(series)
                    y_train.append(self.targets[index])
                else:
                    x_test.append(series)
                    y_test.append(self.targets[index])
        else:
            training_data_len = math.ceil(self.batches.shape[0] * split_ratio)
            x_train, y_train = self.batches[:training_data_len], self.targets[:training_data_len]
            x_test, y_test = self.batches[training_data_len:], self.targets[training_data_len:]

        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

        x_train, y_train = self.balance_targets(x_train, y_train, self.target_counts(x_train, y_train, 'training set'))

        print("Targets after balancing: ")

        self.target_counts(x_train, y_train, 'training set')

        self.target_counts(x_test, y_test, 'testing set')

        self.testing_targets = x_test.copy()
        self.unscaled_targets = y_test.copy()

        return x_train, y_train, x_test, y_test

    # Undersample (Drop series to balance dataset)
    def balance_targets(self, x_train, y_train, training_imbalance):
        num_batches_to_drop = training_imbalance[0] - training_imbalance[1]

        # If no imbalance, return x_train and y_train
        if num_batches_to_drop == 0:
            return x_train, y_train

        buffer_target = []
        for i in range(x_train.shape[0]):
            buffer_data = x_train[i][-1]
            if buffer_data[1] < y_train[i]:
                buffer_target.append(1)
            else:
                buffer_target.append(0)
        buffer_target_np = np.array(buffer_target)

        if num_batches_to_drop > 0:
            mask = (buffer_target_np == 0.0)
            zero_indices = np.where(mask)[0]
            x_train_balanced = np.delete(x_train, zero_indices[:num_batches_to_drop], axis=0)
            y_train_balanced = np.delete(y_train, zero_indices[:num_batches_to_drop], axis=0)
            return np.array(x_train_balanced), np.array(y_train_balanced)
        elif num_batches_to_drop < 0:
            mask = (buffer_target_np == 1.0)
            one_indices = np.where(mask)[0]
            x_train_balanced = np.delete(x_train, one_indices[:abs(num_batches_to_drop)], axis=0)
            y_train_balanced = np.delete(y_train, one_indices[:abs(num_batches_to_drop)], axis=0)
            return np.array(x_train_balanced), np.array(y_train_balanced)

    # Printing Target counts
    def target_counts(self, data, target, flag='training'):
        buffer_target = []
        for i in range(data.shape[0]):
            buffer_data = data[i][-1]
            if buffer_data[1] < target[i]:
                buffer_target.append(1)
            else:
                buffer_target.append(0)
        buffer_target_np = np.array(buffer_target)
        count_0 = np.count_nonzero(buffer_target_np == 0)
        count_1 = np.count_nonzero(buffer_target_np == 1)
        print(f"For {flag} : Number of 0s: {count_0}, number of 1s: {count_1}")

        return [count_0, count_1]

    def custom_scaling(self, x_train, y_train, x_test, y_test, testing=False):

        def scale(index, buffer_series, target, flag):
            continuous_series = buffer_series[:, :self.cv]
            max_series = np.amax(continuous_series)
            min_series = np.amin(continuous_series)
            scaled_seq = []
            if flag != 'training':
                self.scaling_dict[index] = (min_series, max_series)
            for index, j in enumerate(continuous_series):
                buffer_seq = []
                for k in j:
                    new_k = (k - min_series) / (max_series - min_series)
                    buffer_seq.append(new_k)
                buffer_seq = list(buffer_seq) + list(buffer_series[index][self.cv:])
                scaled_seq.append(buffer_seq)
            scaled_target = (target - min_series) / (max_series - min_series)
            return np.array(scaled_seq), scaled_target

        if not testing:
            for i, series in enumerate(x_train):
                x_train[i], y_train[i] = scale(i, series, y_train[i], 'training')

        for i, series in enumerate(x_test):
            x_test[i], y_test[i] = scale(i, series, y_test[i], 'testing')

        print("scaling successful")
        print("type of x_train, y_train: ", type(x_train), type(y_train))
        return x_train, y_train, x_test, y_test

    def inverse_scaling(self, predictions):
        for i, pred in enumerate(predictions):
            min_series, max_series = self.scaling_dict[i]
            predictions[i] = pred * (max_series - min_series) + min_series

        return predictions

    def get_scaled_series_for_prediction(self, buffer_series):
        print("Inside get_scaled_series, cv is: ", self.cv)
        continuous_series = buffer_series[:, :self.cv]
        print("continuous series is: ", continuous_series[-5:])
        max_series = np.amax(continuous_series)
        min_series = np.amin(continuous_series)
        scaled_seq = []
        scaling_tuple = (min_series, max_series)
        for index, j in enumerate(continuous_series):
            buffer_seq = []
            for k in j:
                new_k = (k - min_series) / (max_series - min_series)
                buffer_seq.append(new_k)
            buffer_seq = list(buffer_seq) + list(buffer_series[index][self.cv:])
            scaled_seq.append(buffer_seq)
        return scaling_tuple, np.array(scaled_seq)

    def get_transformed_prediction(self, buffer_prediction, buffer_scaling_tuple):
        min_series, max_series = buffer_scaling_tuple
        return buffer_prediction * (max_series - min_series) + min_series


class Preprocessor(DataImport, TransformData):
    def __init__(self, stock='IBM', cv=4, seq_len=30):
        DataImport.__init__(self, stock)
        TransformData.__init__(self,seq_len, cv)

    def get_data_and_create_features(self, flag='default'):
        print("Inside preprocessor")
        return self.create_features(self.get_data(flag=flag, seq_len=30))

    def create_scaled_and_balanced_batches(self, df, show_sample_scale_example=True):
        return self.transformation_workflow(df, show_sample_scale_example)
