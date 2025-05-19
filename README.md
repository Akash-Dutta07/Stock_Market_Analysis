# 💰 FinSight AI: LSTM-Based Stock Forecasting App

Welcome to **FinSight AI**, a powerful and intuitive Streamlit web application that leverages **Stacked LSTM deep learning models** to forecast stock prices and compute **CAPM-based expected returns and Beta values**. 

Designed for finance students. This project combines machine learning with capital market insights to offer:

- Accurate historical stock price prediction
- CAPM model outputs (returns, beta)
- Error metrics: MAE and RMSE


---

## 📊 Features

### ✅ Overview Page
- Instructions to help you navigate the application.

### 🔍 Capital Asset Pricing Model (CAPM)
- Visualize historical stock returns
- Calculate expected return using CAPM
- Derive Beta values for individual assets

### 📊 Stock Forecasting (with LSTM)
- Trained stacked LSTM models for 4 stocks (AAPL, AMZN, GOOGL, MGM)
- Actual vs predicted price visualization
- Forecast accuracy via MAE and RMSE
- Predict the next 7 or 30 business days' closing price

---

## 📆 Models Trained On
- 5 years of Yahoo Finance historical data
- Each stock's `.keras` model trained & stored under `/Trains`

---

## 📝 Tech Stack

- Python 3.x
- Streamlit
- TensorFlow/Keras
- yFinance
- Matplotlib & pandas
- Scikit-learn

---
