import pandas as pd
import matplotlib.pyplot as plt

class PrintPlotter:
    def __init__(self):
        pass

    def print_model_profitability_metrics(self, metric_dict, number_of_days):
        profit = metric_dict['profit_percentage']
        loss = metric_dict['loss_percentage']
        loss_w_sl = metric_dict['loss_w_sl_percentage']
        tp = metric_dict['tp']
        fp = metric_dict['fp']
        fn = metric_dict['fn']
        tn = metric_dict['tn']
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"The model is accurate {accuracy * 100}% of the time")
        print(f"Number of profitable days: {len(profit)} , loss days: {len(loss)}")
        print(f"Total profit: {sum(profit)}, loss: {sum(loss)}, loss with SL: {sum(loss_w_sl)}")
        print(
            f"Average profit: {sum(profit) / len(profit)}, loss: {sum(loss) / len(loss)}, loss with SL: {sum(loss_w_sl) / len(loss_w_sl)}")
        total_years = round(number_of_days / 252, 1)

        total_months = round(total_years * 12)

        print("Yearly profitability: ", (sum(profit) - sum(loss)) / total_years)
        print("Yearly profitability with SL: ", (sum(profit) - sum(loss_w_sl)) / total_years)
        print("Monthly profitability: ", (sum(profit) - sum(loss)) / total_months)
        print("Monthly profitability with SL: ", (sum(profit) - sum(loss_w_sl)) / total_months)

    def get_monthly_profit(self,buffer_metric_dict, df, plot_profits = True):
        buffer_df = pd.DataFrame()
        buffer_df['Net'] = buffer_metric_dict['net_percentage']
        buffer_df['Net_with_sl'] = buffer_metric_dict['net_percentage_with_sl']
        buffer_df['Date'] = df[-buffer_df.shape[0]:].index

        buffer_test_df = buffer_df[['Date', 'Net', 'Net_with_sl']]

        plot_df = buffer_test_df

        # If your 'Date' column is not in datetime format, convert it:
        plot_df['Date'] = pd.to_datetime(plot_df['Date'])

        # Group the data by month and calculate the total net for each month
        monthly_totals = plot_df.groupby(plot_df['Date'].dt.to_period('M'))['Net'].sum()
        monthly_totals_w_sl = plot_df.groupby(plot_df['Date'].dt.to_period('M'))['Net_with_sl'].sum()

        # Create a bar plot for the monthly totals
        plt.figure(figsize=(20, 10))
        monthly_totals.plot(kind='bar', color='skyblue')
        plt.xlabel('Month')
        plt.ylabel('Total Net')
        plt.title('Total Net per Month')
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.show()

        # Create a bar plot for the monthly totals
        plt.figure(figsize=(20, 10))
        monthly_totals_w_sl.plot(kind='bar', color='skyblue')
        plt.xlabel('Month')
        plt.ylabel('Total Net')
        plt.title('Total Net per Month')
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.show()

class Testing:
    def __init__(self):
        self.printPlotter = PrintPlotter()


    def evaluate_and_preds(self, model, x_test, y_test):
        return model.evaluate(x_test, y_test), model.predict(x_test)

    def transform_preds(self, buffer_preprocessor, buffer_predictions):
        return buffer_preprocessor.transformData.inverse_scaling(buffer_predictions)

    def combine_predictions_with_actuals(self, buffer_preprocessor, buffer_transformed_preds):
        closing_prices = []

        testing_data = buffer_preprocessor.transformData.testing_targets

        actual_close_price = buffer_preprocessor.transformData.unscaled_targets

        for series in testing_data:
            closing_prices.append(series[-1][1])

        preds, prev_close, actual_close = pd.DataFrame(buffer_transformed_preds), pd.DataFrame(
            closing_prices), pd.DataFrame(actual_close_price)
        dataframe = pd.concat([prev_close, actual_close, preds], axis=1)
        dataframe.columns = ['prev_close', 'actual_close', 'predictions']

        return dataframe

    def calculate_profit_or_loss_percentage(self, buffer_df):
        profit_percentage = []
        loss_percentage = []
        loss_w_sl_percentage = []  # Not sure about the exact purpose from provided context, initializing as an empty list
        net_percentage = []
        net_percentage_with_sl = []

        tp, fp, fn, tn = 0, 0, 0, 0

        for index, row in buffer_df.iterrows():
            prev_close, actual_close, predictions = row['prev_close'], row['actual_close'], row['predictions']
            if predictions > prev_close and actual_close > prev_close:
                daily_profit_perc = min((actual_close - prev_close) / prev_close * 100, (predictions - prev_close) / prev_close * 100)
                profit_percentage.append(daily_profit_perc)
                net_percentage.append(daily_profit_perc)
                net_percentage_with_sl.append(daily_profit_perc)
                tp += 1
            elif predictions < prev_close and actual_close < prev_close:
                daily_profit_perc = min((prev_close - actual_close) / prev_close * 100, (prev_close - predictions) / prev_close * 100)
                profit_percentage.append(daily_profit_perc)
                net_percentage.append(daily_profit_perc)
                net_percentage_with_sl.append(daily_profit_perc)
                tn += 1
            elif predictions > prev_close > actual_close:
                daily_loss_perc = (prev_close - actual_close) / prev_close * 100
                loss_percentage.append(daily_loss_perc)
                loss_w_sl_percentage.append(min(daily_loss_perc, prev_close * 1/100))
                net_percentage.append(-daily_loss_perc)
                net_percentage_with_sl.append(-min(daily_loss_perc, prev_close * 1/100))
                fp += 1
            elif predictions < prev_close < actual_close:
                daily_loss_perc = (actual_close - prev_close) / prev_close * 100
                loss_percentage.append(daily_loss_perc)
                loss_w_sl_percentage.append(min(daily_loss_perc, prev_close * 1/100))
                net_percentage.append(-daily_loss_perc)
                net_percentage_with_sl.append(-min(daily_loss_perc, prev_close * 1/100))
                fn += 1

        metric_dict = {
            'profit_percentage': profit_percentage,
            'loss_percentage': loss_percentage,
            'loss_w_sl_percentage': loss_w_sl_percentage,
            'net_percentage': net_percentage,
            'net_percentage_with_sl': net_percentage_with_sl,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

        return metric_dict

    def profitability(self, buffer_df, print_details = True):
        print("Inside profitability")
        metric_dict = self.calculate_profit_or_loss_percentage(buffer_df)


        if print_details:
            self.printPlotter.print_model_profitability_metrics(metric_dict, buffer_df.shape[0])

        return metric_dict

    def plot_monthly_profit(self, buffer_metric_dict, df, plot_profits = True):
        self.printPlotter.get_monthly_profit(buffer_metric_dict, df, plot_profits = True)
