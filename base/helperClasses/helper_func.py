from .new_preprocessor import Preprocessor
import numpy as np


class HelperFunc:
    def __init__(self):
        pass

    @staticmethod
    def check_if_nse_stock(request):
        if request.data["us_stock"]:
            return request.data["symbol"]
        else:
            return request.data["symbol"] + ".NS"

    @staticmethod
    def predict(stock_name, model):
        preprocessor = Preprocessor(stock=stock_name)
        buffer_series = preprocessor.get_data_and_create_features(flag='predict')
        print("buffer_series is: ", buffer_series.shape, type(buffer_series))
        scaling_tuple, scaled_seq = preprocessor.get_scaled_series_for_prediction(
            np.array(buffer_series[['Open', 'Close']]))
        print("scaling tuple, scaling seq: ", scaling_tuple, type(scaled_seq))
        reshaped_scaled_seq = np.reshape(scaled_seq, (1, scaled_seq.shape[0], scaled_seq.shape[1]))
        prediction = model.predict(reshaped_scaled_seq)
        print("prediction is: ", prediction)
        transformed_prediction = preprocessor.get_transformed_prediction(prediction, scaling_tuple)
        return transformed_prediction, buffer_series['Close'][-1]

    @staticmethod
    def get_model_results_for_stock(stock_name, model):
        preprocessor = Preprocessor(stock=stock_name)
        buffer_series = preprocessor.get_data_and_create_features(flag='get_model_results')
        batches, targets = preprocessor.create_batches(np.array(buffer_series[['Open', 'Close']]))
        buffer_targets = targets.copy()
        _, _, x_test, y_test = preprocessor.custom_scaling(0, 0, batches, targets, testing=True)

        predictions = model.predict(x_test)

        transformed_predictions = preprocessor.inverse_scaling(predictions)
        transformed_predictions_list = transformed_predictions.flatten()
        date_array = list(buffer_series.index[-transformed_predictions_list.shape[0]:])

        return list(transformed_predictions_list), list(buffer_targets), date_array
