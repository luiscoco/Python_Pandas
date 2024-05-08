import pandas as pd


def main():
    # Creating a MultiIndex DataFrame
    arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=('First', 'Second'))
    df_multiindex = pd.DataFrame({'Values': [1, 2, 3, 4]}, index=multi_index)
    print("Original DataFrame with MultiIndex:")
    print(df_multiindex)

    # Aggregating data at different levels of MultiIndex
    level_mean = df_multiindex.groupby(level='First').mean()
    print("\nMean of values grouped by the 'First' level of the MultiIndex:")
    print(level_mean)


if __name__ == "__main__":
    main()
