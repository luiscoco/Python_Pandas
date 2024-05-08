import pandas as pd

# Function to create a sample CSV file


def create_sample_csv():
    data = {
        'Date': ['2023-05-01', '2023-05-01', '2023-05-02', '2023-05-02'],
        'Category': ['A', 'B', 'A', 'B'],
        'Value': [100, 200, 150, 250],
        'ID': [1, 1, 2, 2]
    }
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)

# Function to read data and perform pivoting and melting


def process_data():
    df = pd.read_csv('sample_data.csv')

    # Pivoting
    pivot_table = df.pivot_table(
        index='Date', columns='Category', values='Value')
    print("Pivot Table:")
    print(pivot_table)

    # Melting - revised to melt the DataFrame properly
    melted_df = pd.melt(df, id_vars=['ID', 'Date'], value_vars=[
                        'Value'], var_name='Category', value_name='NumericValue')
    print("\nMelted DataFrame:")
    print(melted_df)

    # Save the pivot table and melted DataFrame to CSV
    pivot_table.to_csv('pivot_table.csv')
    melted_df.to_csv('melted_data.csv')



def main():
    create_sample_csv()
    process_data()


if __name__ == "__main__":
    main()
