# 📈 Crypto Time Series Analysis & Bitcoin Price Forecasting

This project focuses on **analyzing and forecasting Bitcoin prices** using different **Time Series and Machine Learning models**.  
The goal is to understand historical cryptocurrency price patterns and predict future trends using both **statistical and deep learning techniques**.

---

## 🚀 Live Demo

🔗 https://crypto-time-series-analysis.streamlit.app/

---

## 🚀 Project Overview

Cryptocurrency markets are highly dynamic and unpredictable.  
In this project, historical **Bitcoin price data** is analyzed to build forecasting models that can predict future price movements.

The project includes:

- Data preprocessing and analysis  
- Time series forecasting models  
- Model comparison and evaluation  
- Visualization of predictions  

---

## 📊 Dataset

The dataset contains historical **Bitcoin market data** with the following features:

- **Open** – Opening price  
- **High** – Highest price  
- **Low** – Lowest price  
- **Close** – Closing price  
- **Volume** – Trading volume  

---

## 🤖 Models Used

### 1️⃣ ARIMA  
Statistical model for forecasting using past values.

### 2️⃣ SARIMA  
Captures **seasonality** in time series data.

### 3️⃣ Prophet  
Handles **trend + seasonality**, developed by Meta.

### 4️⃣ LSTM  
Deep learning model for **sequence prediction**.

---

## 📏 Model Evaluation Metrics

- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **MAPE (Mean Absolute Percentage Error)**  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Statsmodels  
- Prophet  
- TensorFlow / Keras  
- Streamlit  
- Git & GitHub  

---

## ▶️ How to Run

```bash
git clone https://github.com/Rachanashivarkar/crypto-time-series-analysis.git
cd crypto-time-series-analysis/dashboard
pip install -r requirements.txt
streamlit run app.py