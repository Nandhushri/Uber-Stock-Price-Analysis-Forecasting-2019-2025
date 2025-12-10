# Main Python file for my project
# This file can run the analysis or call other modules
###Load & Explore Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Loading data into dataframe df
df = pd.read_csv('/Users/nandhu/Downloads/uber_stock_data.csv') 
#date into datatime format
df['Date'] = pd.to_datetime(df['Date'])
#Sets date as index 
df.set_index('Date', inplace=True) 
#Sorts the data by date
df.sort_index(inplace=True) 

#For preview
df.head() 
df = df.asfreq('D')  # Set to daily frequency (It has a row for every single day)
df['Close'] = df['Close'].interpolate() #Fills in any missing 'Close' values using linear interpolation.

#Fills any missing 'Volume' values by carrying forward the previous value.
df['Volume'] = df['Volume'].ffill()
#  3. EDA: Stock Performance Analysis

## 3.1 Plot Closing Price Over Time
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title('Uber Closing Stock Price (2019–2025)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()
## 3.2 Highest and Lowest Closing Prices
max_price = df['Close'].max()
min_price = df['Close'].min()
print(f"🔼 Highest Close: ${max_price:.2f} on {df['Close'].idxmax().date()}")
print(f"🔽 Lowest Close:  ${min_price:.2f} on {df['Close'].idxmin().date()}")
## 3.3 Trading Volume Over Time
plt.figure(figsize=(14,4))
plt.plot(df['Volume'], alpha=0.5, label='Daily Volume')
plt.plot(df['Volume'].rolling(30).mean(), label='30-Day Avg Volume', color='orange')
plt.title('Uber Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.legend()
plt.show()

# Optionally save for Power BI
monthly_avg.to_csv('monthly_avg_price.csv')
## 3.4 Monthly Average Closing Price
monthly_avg = df['Close'].resample('ME').mean()

plt.figure(figsize=(14,6))
plt.plot(monthly_avg, marker='o', linestyle='--', label='Monthly Avg Close')
plt.title('Monthly Average Closing Price')
plt.xlabel('Month')
plt.ylabel('Avg Price ($)')
plt.grid(True)
plt.legend()
plt.show()

# Optionally save for Power BI
monthly_avg.to_csv('monthly_avg_price.csv')
#pip install pmdarima
#conda install -c conda-forge pmdarima
from pmdarima import auto_arima
import matplotlib.pyplot as plt
# Fit auto ARIMA model
stepwise_model = auto_arima(df['Close'], start_p=1, start_q=1, max_p=5, max_q=5, d=1, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
# Summary
print(stepwise_model.summary())
# Forecast future values (180 days)
n_periods = 180
forecast_auto, conf_int = stepwise_model.predict(n_periods=n_periods, return_conf_int=True)
# Forecast index
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_periods)
# Plot
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Historical')
plt.plot(forecast_index, forecast_auto, label='Forecast', color='red')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.title('Uber Stock Price Forecast (Auto ARIMA)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()
# 5. Model Training & Evaluation
# ARIMA Model (ARIMA(0, 1, 0) was selected by auto_arima)

# Set up train-test split for a realistic evaluation
train_size = int(len(df) * 0.95)
train, test = df[:train_size], df[train_size:]

# Separate actual values for the test period (for comparison later)
test_values = test['Close']

# Re-fit the ARIMA(0, 1, 0) model on the training data only
from statsmodels.tsa.arima.model import ARIMA

# Fit model
model_arima = ARIMA(train['Close'], order=(0, 1, 0))
model_arima_fit = model_arima.fit()

# Forecast on the test set
forecast_arima = model_arima_fit.forecast(steps=len(test))

# Create a DataFrame for test set forecast
forecast_index_arima = test.index
forecast_arima = pd.Series(forecast_arima, index=forecast_index_arima)
# Plot ARIMA Test Set Forecast
plt.figure(figsize=(14,6))
plt.plot(train['Close'], label='Training Data')
plt.plot(test['Close'], label='Actual Test Data', color='orange')
plt.plot(forecast_arima, label='ARIMA Forecast', color='red')
plt.title('ARIMA(0,1,0) Model Forecast vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['Close'])
plot_pacf(df['Close'])
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# forecast_arima contains your predicted values
# test contains the actual values for the same period
rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))
mae_arima = mean_absolute_error(test, forecast_arima)
print(f"ARIMA RMSE: {rmse_arima:.2f}")
print(f"ARIMA MAE: {mae_arima:.2f}")
#conda install -c conda-forge prophet
# Import necessary libraries
import pandas as pd
import numpy as np
# Load the Uber stock dataset
df = pd.read_csv('/Users/nandhu/Downloads/uber_stock_data.csv') 
# Convert 'Date' to datetime and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
# Resample to daily frequency
df = df.asfreq('D')
df['Close'] = df['Close'].interpolate()
df['Volume'] = df['Volume'].ffill()
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
import matplotlib.pyplot as plt
# Prepare for Prophet
df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
# Initialize and fit model
model = Prophet(daily_seasonality=True)
model.fit(df_prophet)
# Create future dates
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)
# Plot forecast
model.plot(forecast)
plt.title('Uber Stock Forecast - Prophet')
plt.grid(True)
plt.show()
# Plot seasonality/trend components
model.plot_components(forecast)
plt.show()
# Optional: Export forecast
forecast[['ds', 'yhat']].to_csv('prophet_forecast.csv', index=False)
# 6. Prophet Model Evaluation (Test Set)

# Set up train-test split for evaluation (last 30 days)
test_days = 30
train_prophet = df_prophet[:-test_days]
test_prophet = df_prophet[-test_days:]
# Re-fit Prophet model on the training data only
model_test = Prophet(daily_seasonality=True)
model_test.fit(train_prophet)

# Make a dataframe for the test period predictions
future_test = model_test.make_future_dataframe(periods=test_days)
forecast_test = model_test.predict(future_test)
# Only the predicted 'yhat' values for the last 30 days
forecast_prophet = forecast_test[['ds', 'yhat']].set_index('ds').loc[test_prophet['ds']]
# Actual test values
y_true = test_prophet.set_index('ds')['y']
y_pred = forecast_prophet['yhat']
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Calculate RMSE and MAE for Prophet
rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred))
mae_prophet = mean_absolute_error(y_true, y_pred)
print(f"Prophet RMSE: {rmse_prophet:.2f}")
print(f"Prophet MAE: {mae_prophet:.2f}")
# 7. Model Comparison
print("Comparison of Model Performance:")
print(f"ARIMA → RMSE: {rmse_arima:.2f}, MAE: {mae_arima:.2f}")
print(f"Prophet → RMSE: {rmse_prophet:.2f}, MAE: {mae_prophet:.2f}")
