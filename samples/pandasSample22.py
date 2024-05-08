import pandas as pd
from scipy.stats import zscore

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def detect_and_remove_outliers(df, column_name):
    """Detect and remove outliers in a DataFrame based on the z-score."""
    # Calculating z-scores of the data
    df['z_score'] = zscore(df[column_name])

    # Detecting outliers
    outliers = df[df['z_score'].abs() > 3]

    # Removing outliers
    # Use .copy() to explicitly make a copy
    clean_data = df[df['z_score'].abs() <= 3].copy()
    # Now it's safe to modify in-place
    clean_data.drop(columns=['z_score'], inplace=True)

    return outliers, clean_data

def save_data(df, file_path):
    """Save the DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# Main execution block
if __name__ == "__main__":
    # Load the data
    data_path = 'SampleData7.csv'  # specify your data file path
    data = load_data(data_path)

    # Check and remove outliers from a specific column
    outliers, cleaned_data = detect_and_remove_outliers(data, 'Values')

    # Save the cleaned data
    save_data(cleaned_data, 'cleaned_data.csv')
    print("Outliers removed and cleaned data saved.")
