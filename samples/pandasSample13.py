import pandas as pd

def main():
    # Sample data creation
    data = {'Timestamp': ['2023-05-08 14:30',
                          '2023-05-08 15:45', '2023-05-08 16:00']}
    df = pd.DataFrame(data)

    # Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Localizing datetimes to UTC
    localized_dt = df['Timestamp'].dt.tz_localize('UTC')
    print("Localized to UTC:")
    print(localized_dt)

    # Converting time zones to America/New_York
    converted_dt = localized_dt.dt.tz_convert('America/New_York')
    print("\nConverted to America/New_York Timezone:")
    print(converted_dt)


if __name__ == "__main__":
    main()
