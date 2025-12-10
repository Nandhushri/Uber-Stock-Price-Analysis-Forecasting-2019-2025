# Uber Stock Price Analysis & Forecasting

A complete end-to-end data analysis and forecasting project using Uber Technologies’ historical stock prices. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, and time-series forecasting using both **ARIMA** and **Facebook Prophet** models.

---

## Table of Contents

Project Overview

Dataset Description

Project Structure

Tools and Technologies

Key Analysis Steps

Models Used & Performance

Installation

How to Run This Project

Future Improvements

---

##  Project Overview

This project analyzes **Uber stock price data** to understand historical trends, volatility, and price movement patterns.
It then applies two powerful forecasting models — **ARIMA** and **Prophet** — to predict future closing prices and compare performance using standard error metrics.

The project is designed to demonstrate real-world skills in:

* Time-series preprocessing
* Visualization
* Forecasting
* Model evaluation
* Creating production-ready project structure

---

##  Dataset Description

The dataset used in this project is the Uber Stocks Dataset (2019–2025).
It includes historical daily stock data from May 10, 2019, to February 5, 2025.
The dataset contains the following fields:

• Date
• Open price
• High price
• Low price
• Close price
• Adjusted close price
• Volume

This dataset is suitable for exploratory data analysis, visualization, stock trend detection, and basic time-series modeling.



##  Project Structure

```
Uber-Stock-Analysis/
│
├── data/
│   └── uber_stock_data.csv
│
├── notebooks/
│   └── Uber-Stock-Analysis.ipynb
│
├── forecasts/
│   ├── prophet_forecast.csv
│   ├── arima_forecast.csv
│   ├── monthly_avg_price.csv
│   └── comparison_metrics.txt
│
├── README.md
└── requirements.txt
```

---

## Tools and Technologies

• Python
• Pandas
• NumPy
• Matplotlib
• Seaborn
• Jupyter Notebook


##  Key Analysis Steps

### **1. Data Cleaning**

* Parsing dates
* Sorting values
* Handling missing data
* Converting to a time-series index

### **2. Exploratory Data Analysis (EDA)**

* Line charts of closing prices
* Volume trends
* Moving averages (30, 50, 100 days)
* Monthly resampling
* Visualizing volatility

### **3. Feature Engineering**

* Rolling averages
* Daily returns
* Log returns

### **4. Forecasting**

Forecasting is performed using two approaches:

---

##  Models Used

### ** ARIMA Model**

* Suitable for autoregressive time-series
* Requires stationarity
* Used differencing + p,d,q selection

### ** Facebook Prophet**

* Handles trend + seasonality
* Robust to missing data
* Produces future confidence intervals
* Ideal for business forecasting

---

##  Model Performance

The following metrics were used to compare ARIMA and Prophet:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**

Final results are saved inside:

```
forecasts/comparison_metrics.txt
```

---

##  Installation

Make sure Python 3.8+ is installed.
Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/Uber-Stock-Analysis.git
```

### **2. Navigate into the project**

```bash
cd Uber-Stock-Analysis
```

### **3. Start Jupyter Notebook**

```bash
jupyter notebook
```

### **4. Open the notebook**

```
notebooks/Uber-Stock-Analysis.ipynb
```

Run all cells to generate the forecasts and outputs.

---

##  Outputs

All generated files are stored inside `/forecasts/`:

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `prophet_forecast.csv`   | Prophet model predictions    |
| `arima_forecast.csv`     | ARIMA model predictions      |
| `monthly_avg_price.csv`  | Monthly resampled averages   |
| `comparison_metrics.txt` | ARIMA vs Prophet performance |

---

##  Future Improvements

* Add **LSTM / GRU** deep learning models
* Include **multiple stock tickers** for comparison
* Deploy model using **Streamlit**
* Build an interactive **dashboard** for visualization

---

##  Author

**D S Nandhushri**
Data Science | Machine Learning | Time-Series Analysis

---
