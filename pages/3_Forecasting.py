import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Forecasting", page_icon="📈", layout="wide")
st.title("📈 Stock Forecasting with Stacked LSTM")

# --- Stock mapping ---
stocks = {
    "AAPL": "Trains/AAPL_Train.keras",
    "MGM": "Trains/MGM_Train.keras",
    "AMZN": "Trains/AMZN_Train.keras",
    "GOOGL": "Trains/GOOGL_Train.keras"
}

# --- Select stock ---
stock_choice = st.selectbox("Select a stock to forecast", list(stocks.keys()))

# --- Load Data ---
@st.cache_data
def load_stock_data(stock):
    end = datetime.today()
    start = end - timedelta(days=5 * 365)

    df = yf.download(
        tickers=stock,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by='ticker',
        multi_level_index=False
    )

    return df[['Close']] if 'Close' in df.columns else pd.DataFrame()

df = load_stock_data(stock_choice)

if df.empty:
    st.error("Failed to fetch stock data.")
    st.stop()

# --- Plot data ---
st.subheader(f"{stock_choice} - Last 5 Years Closing Price")
st.line_chart(df)

# --- Load Model ---
model = load_model(stocks[stock_choice])

# --- Scale the data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# --- Prepare test set ---
train_size = int(len(scaled_data) * 0.65)
test_data = scaled_data[train_size - 100:]

x_test = []
for i in range(100, len(test_data)):
    x_test.append(test_data[i - 100:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# --- Predictions ---
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# --- Actual values ---
actual = df['Close'].values[train_size:]
actual = actual[:len(predictions)]

# --- Error Metrics ---
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
st.subheader("📊 Error Metrics")
st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")

# --- Plot actual vs predicted ---
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(actual, label="Actual", color='b')
ax.plot(predictions, label="Predicted", color='orange')
ax.set_title(f"{stock_choice} - Actual vs Predicted Closing Prices")
ax.legend()
st.pyplot(fig)

# --- Forecast Table ---
plot_df = pd.DataFrame({
    'Date': df.index[train_size:train_size + len(predictions)],
    'Actual': actual,
    'Predicted': predictions.flatten()
})
plot_df = plot_df.set_index('Date')

st.subheader("📋 Forecast Table")
st.dataframe(plot_df.round(2))

# --- Future Forecast ---
st.subheader("📈 Future Forecast")
forecast_days = st.radio("Select forecast horizon:", options=[7, 30], index=0)

last_100_days = scaled_data[-100:].reshape(1, 100, 1)
future_predictions = []
input_seq = last_100_days.copy()

for _ in range(forecast_days):
    next_pred = model.predict(input_seq)[0][0]
    future_predictions.append(next_pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close': future_predictions.flatten()
}).set_index('Date')

st.dataframe(future_df.round(2))
