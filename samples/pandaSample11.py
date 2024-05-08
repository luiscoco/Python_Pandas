import pandas as pd


def process_data(input_file, output_file):
    # Load data from CSV
    df = pd.read_csv(input_file)

    # Splitting the 'Name' column into multiple columns
    df['Name_split'] = df['Name'].str.split(' ')

    # Stripping whitespace from the 'Text' column
    df['Text'] = df['Text'].str.strip()

    # Replacing old values with new values in the 'Text' column
    df['Text'] = df['Text'].str.replace('old_value', 'new_value', regex=False)

    # Extracting substrings from the 'Email' column
    email_parts = df['Email'].str.extract(r'(\w+)@([\w.]+)')
    df['Email_username'] = email_parts[0]
    df['Email_domain'] = email_parts[1]

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_csv = 'inputText.csv'  # Name of the input CSV file
    output_csv = 'outputText.csv'  # Name of the output CSV file
    process_data(input_csv, output_csv)
