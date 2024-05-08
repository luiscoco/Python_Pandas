import pandas as pd

def create_multiindex_dataframe():
    # Creating a MultiIndex DataFrame
    arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=('First', 'Second'))
    df_multiindex = pd.DataFrame({'Values': [1, 2, 3, 4]}, index=multi_index)
    return df_multiindex

def access_data(df_multiindex):
    # Accessing data with MultiIndex
    value = df_multiindex.loc[('A', 1), 'Values']
    return value

def main():
    # Creating and displaying the DataFrame
    df_multiindex = create_multiindex_dataframe()
    print("MultiIndex DataFrame:")
    print(df_multiindex)

    # Accessing and displaying a specific value
    value = access_data(df_multiindex)
    print("\nAccessed Value for ('A', 1):", value)

if __name__ == "__main__":
    main()
