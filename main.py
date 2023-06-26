import yfinance as yf
import talib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Parameters
tickers = ['AAPL', 'GOOGL', 'MSFT']
start_date = '2015-01-01'
end_date = '2023-01-01'

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to calculate Alligator indicators
def calculate_alligator(stock_data):
    jaw = talib.SMA(stock_data['Close'], timeperiod=13)
    teeth = talib.SMA(stock_data['Close'], timeperiod=8)
    lips = talib.SMA(stock_data['Close'], timeperiod=5)
    stock_data['Jaw'] = jaw
    stock_data['Teeth'] = teeth
    stock_data['Lips'] = lips
    return stock_data

# Function to add trend feature
def calculate_trend(stock_data):
    stock_data['Trend'] = stock_data['Close'].diff(5)
    return stock_data

# Collecting data and calculating indicators
all_data = {}
for ticker in tickers:
    data = download_stock_data(ticker, start_date, end_date)
    data = calculate_alligator(data)
    data = calculate_trend(data)
    all_data[ticker] = data.dropna()

# Preparing data for ML
X = []
y = []
for ticker, data in all_data.items():
    features = data[['Jaw', 'Teeth', 'Lips', 'Trend']].values
    target = (data['Close'].diff(5) > 0).astype(int).values[5:]
    X.append(features[:-5])
    y.append(target)
X = np.concatenate(X)
y = np.concatenate(y)

# Machine Learning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')
