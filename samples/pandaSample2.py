import pandas as pd

# Reading from CSV
df = pd.read_csv('Book1.csv')

# Selecting a column
ages = df['Age']

# Filtering data
young_people = df[df['Age'] < 30]

print(young_people)

# Merging DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2']})
df2 = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                    'D': ['D0', 'D1', 'D2']})
merged_df = pd.concat([df1, df2], axis=1)

print(merged_df)

# Writing to CSV
merged_df.to_csv('output.csv', index=False)
