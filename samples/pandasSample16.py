import pandas as pd
import json


def read_json_file(file_path):
    """ Reads a JSON file and converts it to a pandas DataFrame. """
    try:
        df = pd.read_json(file_path)
        print("DataFrame from JSON file:")
        print(df)
    except Exception as e:
        print("Error reading JSON file:", e)


def json_string_to_df(json_str):
    """ Converts a JSON string to a pandas DataFrame. """
    try:
        # Convert JSON string to dictionary
        json_dict = json.loads(json_str)
        # Convert dictionary to DataFrame
        df = pd.DataFrame([json_dict])
        print("DataFrame from JSON string:")
        print(df)
    except Exception as e:
        print("Error converting JSON string to DataFrame:", e)


if __name__ == "__main__":
    # Path to the JSON file
    json_file_path = 'data.json'
    # JSON string
    json_string = '{"Name": "Alice", "Age": 30}'

    # Read JSON from file
    read_json_file(json_file_path)

    # Convert JSON string to DataFrame
    json_string_to_df(json_string)
