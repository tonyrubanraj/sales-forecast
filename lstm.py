# Importing the libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# LSTM class to initiatilize variables and perform preprocessing, model training and prediction using different methods on offer
class LSTM(object):
    calendar_df = None
    sales_df = None
    scaler = None

    def __init__(self, calendar_df, sales_df) -> None:
        self.sales_df = sales_df
        self.calendar_df = calendar_df

    def preprocess(self, timesteps = 14):
        """
        Method to preprocess the dataset for the LSTM model

        Arguments:
        timesteps -> the number of days to be used to calculate the next day sales

        Return:
        X_train -> training dataset
        y_train -> expected sales data for the training dataset
        X_test -> test dataset
        daysBeforeEventTest -> a list containing if the previous day had any special event
        """

        # grouping the sales data by store ids
        store_sales_df = self.sales_df.groupby(['store_id']).sum().T

        # training dataset
        training_dataset = store_sales_df[:1880]

        # creating a column to identify if the previous date was a holiday
        daysBeforeEvent = pd.DataFrame(np.zeros((1941,1)))
        for x,y in self.calendar_df.iterrows():
            if((pd.isnull(self.calendar_df["event_name_1"][x])) == False):
                daysBeforeEvent[0][x-1] = 1 

        # "daysBeforeEventTest" will be used as input for predicting (We will forecast the days 1880 - 1941)
        daysBeforeEventTest = daysBeforeEvent[1880:]

        # "daysBeforeEvent" will be used for training as a feature.
        daysBeforeEvent = daysBeforeEvent[:1880]

        # before concatanation with the main data "training_dataset", indexes are made same and column name is changed to "oneDayBeforeEvent"
        daysBeforeEvent.columns = ["oneDayBeforeEvent"]
        daysBeforeEvent.index = training_dataset.index

        #  combine the training data with the newly created event column data
        training_dataset = pd.concat([training_dataset, daysBeforeEvent], axis = 1)

        # scaling the features using min-max scaler
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        training_dataset_scaled = self.scaler.fit_transform(training_dataset)

        # creating X and y datasets to train the model
        X_train = []
        y_train = []
        for i in range(timesteps, 1880):
            X_train.append(training_dataset_scaled[i-timesteps:i])
            y_train.append(training_dataset_scaled[i][0:10]) 

        del training_dataset_scaled

        # converting the X_train and y_train data to np array to match the inputs to the LSTM model
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = store_sales_df[1880:]

        return (X_train, y_train, X_test, daysBeforeEventTest)

    def model(self, X_train, y_train, epochs = 10, batch_size = 40):
        """
        Method to train the LSTM model

        Arguments:
        X_train -> training dataset
        y_train -> expected sales data for the training dataset
        epochs -> Number of iterations that model had to train the dataset
        batch_size -> the number of records to be grouped before training the model

        Return:
        regressor -> the trained LSTM model
        """

        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialising the RNN
        self.regressor = Sequential()

        # Adding the first LSTM layer and Dropout regularisation
        layer_1_units=50
        self.regressor.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        self.regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and Dropout regularisation
        layer_3_units=400
        self.regressor.add(LSTM(units = layer_3_units, return_sequences = True))
        self.regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and Dropout regularisation
        layer_3_units=400
        self.regressor.add(LSTM(units = layer_3_units))
        self.regressor.add(Dropout(0.2))

        # Adding the output layer
        self.regressor.add(Dense(units = 10))

        # Compiling the model
        self.regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # Fitting the model to the Training set
        self.regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        
        return self.regressor

    def predict(self, inputs, timesteps, targetDays, daysBeforeEventTest):
        """
        Method to forecast the data

        Arguments:
        inputs -> test dataset
        timesteps -> the number of days of sales data to be considered in predicting the next day sales data
        targetDays -> Number of days for which the sales data needs to be predicted
        daysBeforeEventTest -> a list containing if the previous day had any special event

        Return:
        predictions -> predicted sales data
        """

        X_test = []
        X_test.append(inputs[:])
        X_test = np.array(X_test)
        predictions = []

        # predicting the sales for the next "targetDays" days 
        for j in range(timesteps,timesteps + targetDays):
            predicted_sales = self.regressor.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 11))
            testInput = np.column_stack((np.array(predicted_sales), daysBeforeEventTest[0][1880 + j - timesteps]))
            X_test = np.append(X_test, testInput).reshape(1,j + 1,11)
            predicted_sales = self.scaler.inverse_transform(testInput)[:,0:10]
            predicted_sales = np.array(predicted_sales).ravel()
            predictions.append(predicted_sales)

        return predictions

    def calculateMetrics(self, y_pred, y_actual, store_ids):
        """
        Method to calculate the RMSE and plot the actual vs predicted sales data

        Arguments:
        y_pred -> predicted sales data
        y_actual -> actual sales data for the days
        store_ids -> the ids of the stores used in the dataset

        Return:
        --
        """

        y_pred = np.array(y_pred)
        total_error = 0

        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("LSTM model performance in forecasting next 61 days sales", fontsize=18, y=0.95)
        
        for idx in range(y_pred.shape[1]):
            store = store_ids[idx]
            # calculating total RMSE
            total_rmse = mean_squared_error(y_pred[:, idx], y_actual.iloc[:, idx], squared=False)
            rmse_per_day = total_rmse / 61
            total_error += rmse_per_day
            ax = plt.subplot(5, 2, idx + 1)
            ax.plot(y_pred[:,idx], label = 'Predicted')
            ax.plot(np.array(y_actual.iloc[:,idx]), label = 'Actual Sales')
            ax.legend(loc = 'upper left')
            ax.set_title('Store id : ' + store)
            print("Average RMSE in store " + store + ' : ' + str(rmse_per_day))

        # calculating the average RMSE across the 10 stores per day
        avg_error = total_error / 10
        print("Average RMSE in a store per day : ", avg_error)
        plt.show()