import tkinter as tk
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def load_data():
    file_path = filedialog.askopenfilename()
    if file_path:
        global df
        df = pd.read_csv(file_path)
        print("Data loaded successfully")
        print(df.head())

def plot_correlation_matrix():
    if df is not None:
        plt.figure(figsize=(10, 7))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

def select_column_and_decompose():
    column = column_entry.get()
    if column in df.columns:
        time_series = pd.to_numeric(df[column], errors='coerce').dropna()
        decomposition = seasonal_decompose(
            time_series, model='additive', period=int(period_entry.get()))
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        plt.show()
    else:
        print("Column not found in the dataset")

root = tk.Tk()
root.title("Data Visualization App")

df = None

# Buttons and entry fields for the GUI
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.pack()

correlation_button = tk.Button(
    root, text="Plot Correlation Matrix", command=plot_correlation_matrix)
correlation_button.pack()

column_label = tk.Label(root, text="Enter Column Name for Time Series:")
column_label.pack()
column_entry = tk.Entry(root)
column_entry.pack()

period_label = tk.Label(root, text="Enter Period for Decomposition:")
period_label.pack()
period_entry = tk.Entry(root)
period_entry.pack()

decompose_button = tk.Button(
    root, text="Decompose Time Series", command=select_column_and_decompose)
decompose_button.pack()

root.mainloop()
