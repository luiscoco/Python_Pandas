import pandas as pd

# Function to process each chunk

def process_chunk(chunk):
    # Placeholder for processing logic. Modify this according to your needs.
    # For example, you might want to print the first few rows of each chunk:
    print(chunk.head())

def read_and_process_in_chunks(file_path, chunk_size):
    # Reading the CSV file in chunks
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)

    for chunk in chunk_iter:
        # Optionally, optimize memory usage for specific columns
        if 'Category' in chunk.columns:
            chunk['Category'] = chunk['Category'].astype('category')

        # Process each chunk
        process_chunk(chunk)

if __name__ == "__main__":
    # File path for the large dataset
    file_path = 'large_data.csv'

    # Size of each chunk
    chunk_size = 100000

    # Function call to read and process the file
    read_and_process_in_chunks(file_path, chunk_size)
