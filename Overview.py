import streamlit as st

# Page setup
st.set_page_config(page_title="FinSight AI Analysis", page_icon="🪙", layout="wide")

# Title and Introduction
st.title("📊 Welcome to FinSight AI")
st.markdown("---")

# Hero Section
st.markdown("""
## 🚀 FinSight AI – Smarter CAPM Analytics and Stock Forecasting

This all-in-one financial toolkit allows you to:
- Analyze **asset returns** and **market risk**
- Calculate **Beta** and **CAPM-based expected returns**
- Forecast future **stock prices** using **Stacked LSTM neural networks**

---
""")

# Navigation Instructions
st.markdown("""
### 🧭 How to Navigate
Use the sidebar to explore the modules:

- 📈 **Capital Asset Pricing Model** – Analyze historical performance and CAPM implications  
- 🧮 **Calculate Beta** – Estimate beta values and expected returns  
- 🔮 **Forecasting** – Predict future prices using advanced LSTM models with MAE/RMSE metrics and future projections

---
""")

# About Section
st.markdown("""
### 🔧 Technologies Behind FinSight AI
- **Python**, **NumPy**, **Pandas**, **Matplotlib**
- **Streamlit** for building the UI
- **TensorFlow/Keras** for deep learning-based time series forecasting
- **yFinance** API to fetch historical market data

---
""")

# Footer
st.markdown("""📓
Built by Akash Dutta  
For learning and exploring 💹.
""")
