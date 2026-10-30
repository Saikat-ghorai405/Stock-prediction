import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

if isinstance(google_data.columns, pd.MultiIndex):
  google_data.columns = google_data.columns.get_level_values(0)

model = load_model("Latest_stock_price_model.h5", compile=False)
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)


# ---------------------------------------------------------------
# FUTURE FORECAST: predict the next N days
# ---------------------------------------------------------------
st.subheader("Future Price Forecast")
forecast_days = st.slider("How many days to predict?", 1, 100, 7)

# Scale ALL close prices (same as in the training notebook)
full_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_full = full_scaler.fit_transform(google_data[['Close']])

# Start with the last 100 real days
window = scaled_full[-100:].reshape(1, 100, 1)

future_scaled = []
for _ in range(forecast_days):
    next_val = model.predict(window, verbose=0)[0, 0]      # predict next day
    future_scaled.append(next_val)
    # drop the oldest day, add the new prediction, repeat
    window = np.append(window[:, 1:, :], [[[next_val]]], axis=1)

future_prices = full_scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).reshape(-1)

# Next business days (skips Saturday/Sunday; does not account for market holidays)
future_dates = pd.bdate_range(google_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

forecast_df = pd.DataFrame({'Predicted_Close': future_prices}, index=future_dates)
st.write(forecast_df)
