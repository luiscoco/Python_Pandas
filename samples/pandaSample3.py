import pandas as pd

# Reading from CSV
df = pd.read_csv('Book1.csv')

# Selecting a column
ages = df['Age']

# Calculating mean
mean_age = df['Age'].mean()

# Calculating correlation
correlation = df.corr()

print('Mean_age:', mean_age)  # corrected print statement

print('Correlation: ', correlation)

