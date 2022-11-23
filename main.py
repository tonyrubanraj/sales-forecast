# importing packages
import pandas as pd
from eda import *
from arima import *
from lstm import *
from dashboard import *

# loading the datasets
calendar_df = pd.read_csv("dataset/calendar.csv")
sales_df = pd.read_csv("dataset/sales_train_evaluation.csv")

# calling method to perform eda on calendar df and sales df
eda(calendar_df, sales_df)

# train and predict sales data using ARIMA model
arima = ARIMA(calendar_df, sales_df)
arima.model()

# train and predict sales data using LSTM model
lstm = LSTM(calendar_df, sales_df)
(X_train, y_train, X_test, daysBeforeEventTest) = lstm.preprocess(timesteps=14)
model = lstm.model(X_train=X_train, y_train=y_train, epochs=1, batch_size=40)
y_pred = lstm.predict(inputs=X_train[-1, :, :], timesteps=14, targetDays=61, daysBeforeEventTest=daysBeforeEventTest)
store_ids = list(set(sales_df['store_id']))
lstm.calculateMetrics(y_pred, X_test, store_ids=store_ids)

# creating dashboard application with the best performing model. In this case, LSTM model
app = createApp(store_ids=store_ids, calendar_df=calendar_df, predictions=y_pred)

# starting the dashboard application server
if __name__ == '__main__':
    app.run_server()
