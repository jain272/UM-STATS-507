import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
def load_data():
    df = pd.read_csv("hf://datasets/misikoff/SPX/^SPX.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    df['Date_Original'] = df['Date']
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    return df

# Perform EDA and Handle Missing Data
def perform_eda_and_handle_missing(df):
    print("Dataset Overview:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nDataset Info:")
    print(df.info())

    # Plot closing price over time
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('SPX Closing Price Over Time')
    plt.legend()
    plt.show()

    return df

# Train-Test Split
def train_test_split(df, test_ratio=0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

# Preprocess Data for Prophet
def preprocess_data_for_prophet(df):
    df_prophet = df.copy().reset_index()
    df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    # Drop unused columns if necessary
    df_prophet = df_prophet[['ds', 'y'] + [col for col in df_prophet.columns if col not in ['ds', 'y']]]

    return df_prophet

# Train Prophet Model
def train_prophet_model(df_prophet):
    model = Prophet()
    model.add_seasonality(name='weekly', period=5, fourier_order=3)

    # Add extra regressors (excluding 'ds' and 'y')
    for col in df_prophet.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)

    model.fit(df_prophet)
    return model

# Make Predictions
def make_predictions(model, df, periods):
    future = model.make_future_dataframe(periods=periods)
    
    # Include regressors in future dataframe
    for col in df.columns:
        if col not in ['ds', 'y']:
            future[col] = pd.concat([df[col], df[col].iloc[-periods:]]).reset_index(drop=True)
            future[col] = future[col].iloc[:len(future)]  # Ensure correct length

    forecast = model.predict(future)
    return forecast

# Plot Results and Compare with Actual Values
def plot_results(model, forecast, test_df):
    plt.figure(figsize=(12,6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
    plt.plot(test_df.index, test_df['Close'], label='Actual', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('SPX Closing Price Prediction vs Actual')
    plt.legend()
    plt.show()

    model.plot_components(forecast)
    plt.show()

    # Merge forecast with actual test values
    forecast_merged = forecast[['ds', 'yhat']].merge(test_df.reset_index()[['Date', 'Close']], left_on='ds', right_on='Date')

    y_true = forecast_merged['Close']
    y_pred = forecast_merged['yhat']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")

# Main Execution
if __name__ == "__main__":
    df = load_data()
    df = perform_eda_and_handle_missing(df)

    train_df, test_df = train_test_split(df)
    df_prophet_train = preprocess_data_for_prophet(train_df)

    model = train_prophet_model(df_prophet_train)
    forecast = make_predictions(model, df_prophet_train, periods=len(test_df))
    plot_results(model, forecast, test_df)