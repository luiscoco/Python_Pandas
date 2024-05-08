import pandas as pd


def convert_to_categorical(csv_input_path, csv_output_path):
    """
    Convert a 'Category' column in a CSV file to categorical data type and
    add a new column with the corresponding category codes.
    
    Parameters:
    csv_input_path (str): Path to the input CSV file.
    csv_output_path (str): Path to save the output CSV file.
    """
    # Load the data
    df = pd.read_csv(csv_input_path)

    # Check if the 'Category' column exists
    if 'Category' not in df.columns:
        raise ValueError("CSV file does not contain a 'Category' column.")

    # Convert 'Category' column to categorical
    df['Category'] = df['Category'].astype('category')

    # Create a new column with category codes
    df['Category_code'] = df['Category'].cat.codes

    # Save the modified DataFrame back to a CSV file
    df.to_csv(csv_output_path, index=False)
    print(f"File saved successfully as {csv_output_path}")


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python this_script.py <input_csv_path> <output_csv_path>")
        return

    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    try:
        convert_to_categorical(input_csv_path, output_csv_path)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
