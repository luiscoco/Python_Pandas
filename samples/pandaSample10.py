import pandas as pd

# Creating sample dataframes
data1 = {'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]}
data2 = {'Key': ['A', 'B', 'D'], 'Value2': [4, 5, 6]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Concatenating DataFrames
# Note: Concatenation combines DataFrames along an axis, default is rows (axis=0)
concatenated_df = pd.concat([df1, df2])
print("Concatenated DataFrame:")
print(concatenated_df)

# Joining DataFrames
# Note: 'join' by default uses the index for joining, here we join on a common column using merge
# for demonstration as join needs setting index or using 'lsuffix' and 'rsuffix' when columns overlap.
joined_df = df1.set_index('Key').join(
    df2.set_index('Key'), lsuffix='_left', rsuffix='_right')
print("\nJoined DataFrame:")
print(joined_df)

# Merging DataFrames
# Note: 'merge' is used for combining DataFrames based on common columns or indices
merged_df = pd.merge(df1, df2, on='Key')
print("\nMerged DataFrame:")
print(merged_df)

# You can run this script in any environment where Python and pandas are installed.
