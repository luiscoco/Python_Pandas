import numpy as np
import pandas as pd

def generate_time_series():
    # Generating time series data
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    time_series = pd.Series(np.random.randn(len(date_range)), index=date_range)
    return time_series

def resample_monthly(time_series):
    # Resampling to monthly frequency
    monthly_data = time_series.resample('ME').mean()
    return monthly_data

def shift_data(time_series):
    # Shifting data by one period
    shifted_data = time_series.shift(1)
    return shifted_data

def calculate_rolling_mean(time_series):
    # Calculating rolling mean over a window of 7 days
    rolling_mean = time_series.rolling(window=7).mean()
    return rolling_mean

if __name__ == "__main__":
    # Generate time series data
    time_series = generate_time_series()

    # Resample data to monthly frequency
    monthly_data = resample_monthly(time_series)

    # Shift the data by one period
    shifted_data = shift_data(time_series)

    # Calculate rolling mean over a 7-day window
    rolling_mean = calculate_rolling_mean(time_series)

    # Print the results
    print("Monthly Data:")
    print(monthly_data)
    print("\nShifted Data:")
    print(shifted_data)
    print("\nRolling Mean (7 days):")
    print(rolling_mean)