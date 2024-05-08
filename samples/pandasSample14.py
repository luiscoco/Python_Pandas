import pandas as pd


def load_and_optimize(path):
    # Load the data
    df = pd.read_csv(path)
    print("Memory usage before optimization:")
    print(df.memory_usage(deep=True))

    # Optimize memory usage by converting 'Category' to 'category' type
    df['Category'] = df['Category'].astype('category')

    print("\nMemory usage after optimization:")
    print(df.memory_usage(deep=True))

    return df


if __name__ == "__main__":
    # Path to the CSV file
    path = 'sampleData.csv'
    optimized_df = load_and_optimize(path)
    print("\nOptimized DataFrame:")
    print(optimized_df)
