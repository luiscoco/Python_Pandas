import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def main():
    # Create a sample DataFrame
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")

    # Converting DataFrame to NumPy array
    numpy_array = df.values
    print("Converted NumPy Array:")
    print(numpy_array)
    print("\n")

    # Converting NumPy array back to DataFrame
    df_from_array = pd.DataFrame(numpy_array, columns=['A', 'B', 'C'])
    print("DataFrame from NumPy Array:")
    print(df_from_array)
    print("\n")

    # Converting DataFrame to SciPy sparse matrix
    sparse_matrix = csr_matrix(df.values)
    print("SciPy Sparse Matrix:")
    print(sparse_matrix)
    print("\n")

if __name__ == "__main__":
    main()
