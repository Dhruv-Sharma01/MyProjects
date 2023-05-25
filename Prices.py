import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score
# Set the ticker symbol for the Nifty50 index
ticker_symbol = '^NSEI'

# Set the start and end dates for the data
start_date = '2016-01-01'
end_date = '2018-07-31'

# Import the historical data using yfinance
data_whole = yf.download(ticker_symbol, start=start_date, end=end_date)

# Print the first few rows of the data


# Load the dataset

data19=yf.download(ticker_symbol, start='2019-01-01', end='2019-12-31')
# Split the dataset into training and testing sets
X = data_whole.drop('Close', axis=1)
X = X.drop('Adj Close', axis=1)
X_19=data19.drop('Close',axis=1)
X_19=X_19.drop('Adj Close',axis=1)
# data['Date']=list(map(float,data['D']))
y = data_whole['Close']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X, y)
y19or=data19['Close']
# Predict the stock prices on the test set
# y_pred = model.predict(X_test)
y_pred19=model.predict(X_19)
r2 = r2_score(y19or, y_pred19)

# print('R-squared score:', r2)
# print(accuracy_score(y19or, y_pred19))
plt.subplot(1,2,1)
plt.plot(np.arange(1,241),y_pred19)
plt.title('Predicted 2019')
plt.subplot(1,2,2)
plt.plot(np.arange(1,241),y19or)
plt.title('Actual 2019')
# plt.title('2019')
plt.show()
# Evaluate the model
print('Hello world',mean_absolute_error(y19or, y_pred19))
# print(accuracy_score(y19or, y_pred19))
print('R-squared score:', r2)

