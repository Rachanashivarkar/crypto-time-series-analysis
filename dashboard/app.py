import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# PAGE SETTINGS

st.set_page_config(page_title="Crypto Time Series Dashboard", layout="wide")

st.title("Bitcoin Time Series Forecasting Dashboard")


# LOAD DATA

data = pd.read_csv("data/processed/bitcoin_processed.csv")
arima_pred = pd.read_csv("results/arima_predictions.csv")
sarima_pred = pd.read_csv("results/sarima_predictions.csv")
prophet_pred = pd.read_csv("results/prophet_predictions.csv")
lstm_pred = pd.read_csv("results/lstm_predictions.csv")


# MOVING AVERAGES

data["MA7"] = data["close"].rolling(7).mean()
data["MA30"] = data["close"].rolling(30).mean()


# KEY STATISTICS

st.subheader("Key Bitcoin Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Price", round(data["close"].mean(),2))
col2.metric("Maximum Price", round(data["close"].max(),2))
col3.metric("Minimum Price", round(data["close"].min(),2))
col4.metric("Total Records", len(data))

st.divider()

# PRICE TREND

st.subheader("📉 Bitcoin Price Trend")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(data["close"], label="Closing Price")
ax.set_title("Bitcoin Closing Price Trend")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)


# MOVING AVERAGE GRAPH

st.subheader("Moving Average Analysis")

recent = data.tail(200)

fig2, ax2 = plt.subplots(figsize=(10,4))

ax2.plot(recent["close"], label="Actual Price")
ax2.plot(recent["MA7"], label="7 Day Moving Average")
ax2.plot(recent["MA30"], label="30 Day Moving Average")

ax2.set_title("Moving Average Trend")
ax2.legend()

st.pyplot(fig2)

st.divider()


# FORECAST COMPARISON

st.subheader("Forecast Comparison")
test = data["close"].tail(68) 
arima_values = arima_pred["prediction"]
sarima_values = sarima_pred["prediction"]
prophet_values = prophet_pred["yhat"].tail(len(test))
lstm_values = lstm_pred["prediction"] 
fig4, ax4 = plt.subplots(figsize=(6,3)) 
ax4.plot(test.values, label="Actual")
ax4.plot(arima_values.values, label="ARIMA")
ax4.plot(sarima_values.values, label="SARIMA")
ax4.plot(prophet_values.values, label="Prophet") 
ax4.plot(range(len(test)-len(lstm_values), len(test)), lstm_values.values, label="LSTM") 
ax4.legend() 
ax4.set_title("Bitcoin Forecast Comparison") 
st.pyplot(fig4) 
st.divider() 

# MODEL PERFORMANCE METRICS

st.subheader("📉 Model Performance Comparison")

metrics = pd.DataFrame({
    "Model": ["ARIMA","SARIMA","Prophet","LSTM"],
    "MAE":[15833.17,28715.41,7447.82,4645.98],
    "RMSE":[19503.58,34121.26,9319.37,5837.67],
    "MAPE":[16.65,29.91,6.83,4.57]
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(metrics)

with col2:
    fig4, ax4 = plt.subplots()
    ax4.bar(metrics["Model"], metrics["RMSE"])
    ax4.set_title("RMSE Comparison")
    ax4.set_ylabel("Error")
    st.pyplot(fig4)

st.divider()


# ERROR COMPARISON GRAPH

st.subheader("📊 Error Metrics Comparison")

fig5, ax5 = plt.subplots()

ax5.plot(metrics["Model"], metrics["MAE"], marker="o", label="MAE")
ax5.plot(metrics["Model"], metrics["RMSE"], marker="o", label="RMSE")

ax5.set_title("Model Error Comparison")
ax5.legend()

st.pyplot(fig5)

st.success("Dashboard Loaded Successfully ✅")