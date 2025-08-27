import pandas as pd
import numpy as np
import sqlite3

# Connecting and extracting data from SQLite databases
conn = sqlite3.connect("data/race_data.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

print(race_df.columns)

race_df.rename(columns={"Time(s)": "TimeSec"}, inplace=True)


for row in race_df.itertuples():

   if row.Status.startswith('+') or row.Status == "Finished" or row.Status == "Lapped":
      if str(row.TimeSec) == 'nan':
         print(row)
