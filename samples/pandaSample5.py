import pandas as pd


def clean_data(input_file, output_file):
    # Read data from CSV
    df = pd.read_csv(input_file)

    # Filling missing values with 0
    df.fillna(0, inplace=True)

    # Dropping rows with missing values
    df.dropna(inplace=True)

    # Convert columns to appropriate types (e.g., numeric)
    df = df.infer_objects()

    # Interpolating missing values for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate()

    # Save cleaned data to a new CSV file
    df.to_csv(output_file, index=False)

    print("Data cleaning completed. Cleaned data saved to", output_file)


def main():
    input_file = 'input_data1.csv'  # Replace with the path to your input CSV file
    output_file = 'cleaned_data1.csv'  # Replace with the desired output CSV file name

    clean_data(input_file, output_file)


if __name__ == "__main__":
    main()
