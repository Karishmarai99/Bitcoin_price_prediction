import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Download Bitcoin data
data = yf.download("BTC-USD", start="2020-01-01", end="2024-01-01")

print(data.head())

# Visualize Bitcoin price
plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.title("Bitcoin Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Prepare dataset
data = data[['Close']]
data['Prediction'] = data['Close'].shift(-30)

# Feature and target
X = np.array(data.drop(['Prediction'], axis=1))
X = X[:-30]

y = np.array(data['Prediction'])
y = y[:-30]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Predict future prices
prediction = model.predict(X[-30:])
print("Future Prediction:", prediction)
print("hello")