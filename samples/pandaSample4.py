import matplotlib.pyplot as plt
import pandas as pd

# Reading from CSV
df = pd.read_csv('Book1.csv')

# Plotting with index as x-axis
df.plot(x=None, y='Age', kind='bar')
plt.show()
