# Python a brifef introduction to Pandas

**Pandas** is a powerful library in Python used for **data manipulation and analysis**

It provides data structures and functions that make working with structured data easy and intuitive

Here's a brief overview of some of its main features along with code samples you can try in VSCode:

**Data Structures**: Pandas primarily deals with three data structures: Series, DataFrame, and Panel

**Series**: A one-dimensional array-like object containing an array of data and an associated array of labels (index)

**DataFrame**: A two-dimensional labeled data structure with columns of potentially different types. It's like a spreadsheet or SQL table

**Panel**: A three-dimensional data structure, but its use is less common compared to Series and DataFrame

```python
import pandas as pd

# Creating a Series
s = pd.Series([1, 3, 5, 7, 9])
print(s)

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df)
```

![image](https://github.com/luiscoco/Python_Pandas/assets/32194879/c57458e2-1e7d-4566-8414-018644d83667)

## 1. Reading and Writing Data:

Pandas can read data from various file formats like CSV, Excel, JSON, SQL databases, etc

It also allows writing data back to these formats

```python
# Reading from CSV
df = pd.read_csv('data.csv')

# Writing to CSV
df.to_csv('output.csv', index=False)
```

## 2. Data Manipulation:

Pandas provides a wide range of functions for data manipulation, including selection, filtering, merging, grouping, sorting, and more

```python
# Selecting a column
ages = df['Age']

# Filtering data
young_people = df[df['Age'] < 30]

# Merging DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})
df2 = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                    'D': ['D0', 'D1', 'D2']})
merged_df = pd.concat([df1, df2], axis=1)
```

## 3. Data Analysis:

Pandas provides statistical and mathematical functions to analyze data, such as mean, median, standard deviation, correlation, etc

```python
# Calculating mean
mean_age = df['Age'].mean()

# Calculating correlation
correlation = df.corr()
```

## 4. Data Visualization Integration:

Pandas integrates well with popular data visualization libraries like Matplotlib and Seaborn, making it easy to visualize data directly from DataFrames

```python
import matplotlib.pyplot as plt

# Plotting
df.plot(x='Name', y='Age', kind='bar')
plt.show()
```


## 5. Handling Missing Data:

Pandas provides methods to handle missing or NaN (Not a Number) values, including filling missing values, dropping rows or columns with missing values, and interpolating missing values

```python
# Filling missing values
df.fillna(0, inplace=True)

# Dropping rows with missing values
df.dropna(inplace=True)

# Interpolating missing values
df.interpolate(inplace=True)
```

## 6. Grouping and Aggregating Data:

Pandas allows you to group data based on one or more keys and perform aggregate functions on each group

```python
# Grouping data
grouped = df.groupby('Category')

# Applying aggregate functions
grouped_mean = grouped.mean()
```

## 7. Reshaping Data:

Pandas provides functions to reshape data, such as pivoting, melting, stacking, and unstacking

```python
# Pivoting
pivot_table = df.pivot_table(index='Date', columns='Category', values='Value')

# Melting
melted_df = pd.melt(df, id_vars=['ID'], value_vars=['A', 'B', 'C'], var_name='Variable', value_name='Value')
```

## 8. Time Series Data Handling:

Pandas has extensive support for working with time series data, including date/time indexing, resampling, shifting, and rolling window calculations

```python
# Creating a time series
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
time_series = pd.Series(range(len(date_range)), index=date_range)

# Resampling
weekly_data = time_series.resample('W').mean()

# Rolling window calculation
rolling_mean = time_series.rolling(window=7).mean()
```

## 9. Categorical Data Handling:

Pandas provides support for categorical data types, which can improve performance and memory usage for certain operations

```python
# Converting a column to categorical
df['Category'] = df['Category'].astype('category')

# Accessing category codes
df['Category_code'] = df['Category'].cat.codes
```

## 10. Combining and Merging Data:

Pandas offers various functions for combining and merging DataFrames, including concatenation, joining, and merging on specific columns or indices

```python
# Concatenating DataFrames
concatenated_df = pd.concat([df1, df2])

# Joining DataFrames
merged_df = df1.join(df2, on='Key')

# Merging DataFrames
merged_df = pd.merge(df1, df2, on='Key')
```

These are some of the advanced features of Pandas that allow you to handle and manipulate data efficiently

Experimenting with these features in VSCode with real-world datasets will enhance your understanding and proficiency with Pandas

## 11. Handling Text Data:

Pandas provides string methods to efficiently manipulate text data in DataFrame columns, such as splitting, stripping, replacing, and extracting substrings

```python
# Splitting text into multiple columns
df['Name'].str.split(' ')

# Stripping whitespace
df['Text'].str.strip()

# Replacing values
df['Text'].str.replace('old_value', 'new_value')

# Extracting substrings
df['Email'].str.extract(r'(\w+)@(\w+)\.com')
```

## 12. Applying Functions Element-wise:

You can apply custom or built-in functions element-wise to DataFrame columns or rows using the apply() method

```python
# Applying a custom function
def square(x):
    return x ** 2

df['Age_squared'] = df['Age'].apply(square)
```

## 13. Handling Time Zones:

Pandas supports time zone handling, allowing you to localize and convert datetimes between different time zones

```python
# Localizing datetimes
localized_dt = df['Timestamp'].dt.tz_localize('UTC')

# Converting time zones
converted_dt = localized_dt.dt.tz_convert('America/New_York')
```

## 14. Memory Optimization:

Pandas provides methods to optimize memory usage, which can be helpful when working with large datasets

```python
# Optimizing memory usage
df_optimized = df.copy()
df_optimized['Category'] = df_optimized['Category'].astype('category')
```

## 15. Handling MultiIndex DataFrames:

Pandas supports hierarchical indexing (MultiIndex), allowing you to work with higher-dimensional data more efficiently

```python
# Creating a MultiIndex DataFrame
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
multi_index = pd.MultiIndex.from_arrays(arrays, names=('First', 'Second'))
df_multiindex = pd.DataFrame({'Values': [1, 2, 3, 4]}, index=multi_index)

# Accessing data with MultiIndex
value = df_multiindex.loc[('A', 1), 'Values']
```

## 16. Working with JSON Data:

Pandas can handle JSON data efficiently, allowing you to read JSON files or convert JSON strings to DataFrame objects

```python
# Reading JSON data
df_json = pd.read_json('data.json')

# Converting JSON string to DataFrame
import json
json_data = '{"Name": "Alice", "Age": 30}'
df_from_json = pd.read_json(json_data, typ='series')
```

## 17. Interoperability with NumPy and SciPy:

Pandas seamlessly interoperates with NumPy and SciPy, allowing you to convert between DataFrame and NumPy array or SciPy sparse matrix

```python
# Converting DataFrame to NumPy array
numpy_array = df.values

# Converting NumPy array to DataFrame
df_from_array = pd.DataFrame(numpy_array, columns=['A', 'B', 'C'])

# Converting DataFrame to SciPy sparse matrix
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(df.values)
```

These advanced features extend the capabilities of Pandas for handling various data manipulation tasks effectively

Experimenting with these features in VSCode with different datasets will enhance your data analysis skills

## 18. Time Series Analysis:

Conducting time series analysis often involves resampling, shifting, and rolling window calculations

```python
# Generating time series data
import numpy as np
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
time_series = pd.Series(np.random.randn(len(date_range)), index=date_range)

# Resampling to monthly frequency
monthly_data = time_series.resample('M').mean()

# Shifting data by a certain number of periods
shifted_data = time_series.shift(1)

# Calculating rolling mean over a window of 7 days
rolling_mean = time_series.rolling(window=7).mean()
```

## 19. Handling Large Datasets:

When working with large datasets, memory optimization and efficient processing become crucial

```python
# Reading large dataset in chunks
chunk_size = 100000
chunk_iter = pd.read_csv('large_data.csv', chunksize=chunk_size)
for chunk in chunk_iter:
    process_chunk(chunk)

# Optimizing memory usage
df['Category'] = df['Category'].astype('category')
```

## 20. Combining Data from Multiple Sources:

Integrating data from different sources and formats can require complex merging and joining operations

```python
# Merging data from multiple CSV files
file1 = pd.read_csv('data1.csv')
file2 = pd.read_csv('data2.csv')
merged_data = pd.merge(file1, file2, on='key')

# Joining data from SQL database
import sqlite3
conn = sqlite3.connect('database.db')
query = "SELECT * FROM table1 JOIN table2 ON table1.key = table2.key"
sql_data = pd.read_sql_query(query, conn)
```

## 21. Handling Hierarchical Data:

Working with hierarchical or nested data structures requires advanced indexing and manipulation techniques

```python
# Creating a MultiIndex DataFrame
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
multi_index = pd.MultiIndex.from_arrays(arrays, names=('First', 'Second'))
df_multiindex = pd.DataFrame({'Values': [1, 2, 3, 4]}, index=multi_index)

# Aggregating data at different levels of MultiIndex
level_mean = df_multiindex.groupby(level='First').mean()
```

## 22. Handling Outliers and Anomalies:

Identifying and handling outliers and anomalies in data requires advanced statistical techniques

```python
# Detecting outliers using z-score
from scipy.stats import zscore
outliers = df[(np.abs(zscore(df['Values'])) > 3)]

# Removing outliers
clean_data = df[(np.abs(zscore(df['Values'])) < 3)]
```

## 23. Advanced Data Visualization:

Visualizing complex relationships in data often requires advanced plotting techniques

```python
# Plotting correlation matrix
import seaborn as sns
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Plotting time series with trend and seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(time_series, model='additive')
decomposition.plot()
```

These examples showcase the versatility of Pandas for handling various data analysis tasks, from time series analysis to data integration and outlier detection

Experimenting with these examples in VSCode with real-world datasets will deepen your understanding of Pandas and enhance your data analysis skills
