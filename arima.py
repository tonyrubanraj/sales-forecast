# Importing the libraries and packages
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# ARIMA class to initiatilize variables and perform preprocessing, model training and prediction using the model method
class ARIMA(object):
    calendar_df = None
    sales_df = None

    def __init__(self, calendar_df, sales_df) -> None:
        self.calendar_df = calendar_df
        self.sales_df = sales_df

    def model(self):
        """
        Method to train the dataset with ARIMA model and make predictions
        """
        
        # grouping the sales information by store  
        store_sales_df = self.sales_df.groupby(['store_id']).sum().T

        # fetching the store ids
        store_ids = list(set(self.sales_df['store_id']))

        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("ARIMA model performance in forecasting next 61 days sales", fontsize=18, y=0.95)
        total_error = 0

        # training and making sales prediction for each store
        for idx in range(len(store_ids)):
            store = store_ids[idx]
            train = np.asarray(store_sales_df[store][0:1880].astype(float))
            test = np.asarray(store_sales_df[store][1880:-1].astype(float))

            # model is trained with the dataset
            model = sm.tsa.statespace.SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,7))
            results = model.fit()

            # plotting the actual vs predicted sales graph
            ax = plt.subplot(5, 2, idx + 1)
            # making predictions
            ax.plot(results.predict(start = len(train), end = len(store_sales_df[store])), label = 'Predicted')
            ax.plot(test, label = 'Actual Sales')
            ax.legend(loc = 'upper left')
            ax.set_title('Store id : ' + store)

            # calculating the total root mean squared error(rmse) and rmse per day
            total_rmse = mean_squared_error(results.predict(start = len(train) + 1, end = len(train) + len(test)), test, squared=False)
            rmse_per_day = total_rmse / (len(test))
            total_error += rmse_per_day
            print("Average RMSE in store " + store + ' : ' + str(rmse_per_day))

        # calculating the average RMSE across the 10 stores per day
        avg_error = total_error/10
        print("Average RMSE in a store per day : ", avg_error)
        plt.show()