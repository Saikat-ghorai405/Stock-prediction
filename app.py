
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# Enter Stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Download stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error("No stock data found. Please enter a valid Stock ID.")
    st.stop()

if isinstance(google_data.columns, pd.MultiIndex):
    google_data.columns = google_data.columns.get_level_values(0)

# Load trained model
model = load_model("Latest_stock_price_model.h5", compile=False)

st.subheader("Stock Data")
st.write(google_data)

# Split data
splitting_len = int(len(google_data) * 0.7)

x_test = pd.DataFrame(google_data.Close[splitting_len:])

# Graph plotting function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(values, color='orange')
    ax.plot(full_data.Close, color='blue')

    if extra_data:
        ax.plot(extra_dataset)

    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend(["Moving Average", "Original Close Price"])

    fig.tight_layout()

    return fig


# MA for 250 days
st.subheader("Original Close Price and MA for 250 days")

google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()

fig = plot_graph((15, 6), google_data['MA_for_250_days'], google_data)
st.pyplot(fig)
plt.close(fig)


# MA for 200 days
st.subheader("Original Close Price and MA for 200 days")

google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()

fig = plot_graph((15, 6), google_data['MA_for_200_days'], google_data)
st.pyplot(fig)
plt.close(fig)


# MA for 100 days
st.subheader("Original Close Price and MA for 100 days")

google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()

fig = plot_graph((15, 6), google_data['MA_for_100_days'], google_data)
st.pyplot(fig)
plt.close(fig)


# MA for 100 days and 250 days
st.subheader("Original Close Price and MA for 100 days and MA for 250 days")

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(google_data['MA_for_100_days'], color='orange')
ax.plot(google_data['MA_for_250_days'], color='green')
ax.plot(google_data.Close, color='blue')

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend(["MA 100 Days", "MA 250 Days", "Original Close Price"])

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(x_test[['Close']])


# Prepare test data
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)


# Model prediction
predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


# Original vs Predicted Data
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

st.subheader("Original Values vs Predicted Values")
st.write(ploting_data)


# Original Close Price vs Predicted Close Price
st.subheader("Original Close Price vs Predicted Close Price")

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    google_data.Close[:splitting_len+100],
    label="Data - Not Used"
)

ax.plot(
    ploting_data['original_test_data'],
    label="Original Test Data"
)

ax.plot(
    ploting_data['predictions'],
    label="Predicted Test Data"
)

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)
