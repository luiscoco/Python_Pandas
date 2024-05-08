import pandas as pd

# Define a custom function to square values

def square(x):
    return x ** 2

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45]
}
df = pd.DataFrame(data)

# Apply the square function to the 'Age' column
df['Age_squared'] = df['Age'].apply(square)

# Display the original DataFrame and the new column
print(df)
