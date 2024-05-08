import pandas as pd


def create_time_series():
    # Creating a time series from January 1, 2024 to December 31, 2024
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    time_series = pd.Series(range(len(date_range)), index=date_range)

    # Resampling the data to weekly averages
    weekly_data = time_series.resample('W').mean()

    # Calculating a 7-day rolling mean
    rolling_mean = time_series.rolling(window=7).mean()

    # Merging all data into a single DataFrame for easier CSV output
    result = pd.DataFrame({
        'Daily': time_series,
        'Weekly Average': weekly_data,
        '7-Day Rolling Mean': rolling_mean
    })

    # Saving the results to a CSV file
    result.to_csv('time_series_data_output.csv')

    print("Data has been processed and saved to 'time_series_data.csv'.")


# Running the function
create_time_series()
