import pandas as pd
import sqlite3

# Merging data from multiple CSV files

def merge_csv_files():
    file1 = pd.read_csv('data1.csv')
    file2 = pd.read_csv('data2.csv')
    merged_data = pd.merge(file1, file2, on='key')
    print("Merged CSV data:")
    print(merged_data)

# Joining data from SQL database

def join_sql_data():
    conn = sqlite3.connect('database.db')
    query = "SELECT * FROM table1 JOIN table2 ON table1.key = table2.key"
    sql_data = pd.read_sql_query(query, conn)
    print("SQL joined data:")
    print(sql_data)

if __name__ == "__main__":
    merge_csv_files()
    join_sql_data()