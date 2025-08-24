# Perform various pandas and numpy functions on your dataset

import pandas as pd
import sqlite3

# Connecting and extracting from Database 
conn = sqlite3.connect("data\\race_data.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

conn = sqlite3.connect("data\\weather_data.db")
weather_df= pd.read_sql("SELECT * FROM weather_table", conn)
conn.close()

print(race_df.tail(20))
print(weather_df.tail(20))