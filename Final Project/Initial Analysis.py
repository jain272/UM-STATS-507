import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

df = pd.read_csv("hf://datasets/misikoff/SPX/^SPX.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Retain the Date column and set it as the index
df['Date_Original'] = df['Date']
df.set_index('Date', inplace=True)
df = df.sort_index()

print("\nShape of dataset:", df.shape)
print("\nSummary Statistics:\n", df.describe())
print("\nDataset Info:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("\n### Missing Values Per Column ###\n", missing_values[missing_values > 0])

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\n### Number of Duplicate Rows: {duplicate_count}")

duplicates_all = df[df.duplicated()]
print(duplicates_all)

# Plot Closing Price over Time with Dates on X-Axis
plt.figure(figsize=(12,6))
plt.plot(df['Date_Original'], df['Close'], label='Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('SPX Closing Price Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Rolling Statistics for Trend Analysis
rolling_mean = df['Close'].rolling(window=50).mean()
rolling_std = df['Close'].rolling(window=50).std()

plt.figure(figsize=(12,6))
plt.plot(df['Date_Original'], df['Close'], label='Original Closing Price', color='blue')
plt.plot(df['Date_Original'], rolling_mean, label='Rolling Mean (50-day)', color='red')
plt.plot(df['Date_Original'], rolling_std, label='Rolling Std (50-day)', color='green')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Rolling Mean & Standard Deviation')
plt.xticks(rotation=45)
plt.legend()
plt.show()