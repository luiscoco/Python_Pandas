import pandas as pd
import traceback


def group_and_calculate_mean(input_file, category_column, value_column, output_file):
    # Read data from CSV
    df = pd.read_csv(input_file)

    # Convert value column to numeric data type, ignoring non-numeric values
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

    # Drop rows with NaN values in the value column
    df.dropna(subset=[value_column], inplace=True)

    # Print datatype after conversion and NaN removal
    print(df[value_column].dtype)  # Should print 'float64'
    print(df.head())  # Print the first few rows to inspect the DataFrame

    # Grouping data by category
    grouped = df.groupby(category_column)

    # Print group data for debugging
    for name, group in grouped:
        print(f"Group: {name}")
        print(group)
        print(group[value_column].dtype)
        if group[value_column].dtype != 'float64':
            print(f"Non-float64 data types found in group {name}")

    # Calculating mean for each group using explicit aggregation
    try:
        grouped_mean = grouped.agg({value_column: 'mean'})
    except Exception as e:
        print("An error occurred while calculating the mean:")
        traceback.print_exc()
        return

    # Save grouped mean data to a new CSV file
    grouped_mean.to_csv(output_file)

    print("Grouping and calculation of mean completed. Grouped mean data saved to", output_file)


def main():
    input_file = 'input_data2.csv'  # Replace with the path to your input CSV file
    category_column = 'Category'  # Replace with the column name you want to group by
    value_column = 'Value'  # Replace with the column name containing numeric values
    # Replace with the desired output CSV file name
    output_file = 'grouped_mean_data.csv'

    group_and_calculate_mean(
        input_file, category_column, value_column, output_file)


if __name__ == "__main__":
    main()
