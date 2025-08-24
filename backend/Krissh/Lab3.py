import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

# Load race data from SQLite database
conn = sqlite3.connect("data\\race_data.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

# Load weather data from SQLite database
conn = sqlite3.connect("data\\weather_data.db")
weather_df = pd.read_sql("SELECT * FROM weather_table", conn)
conn.close()

# Merge race and weather data using RaceID
merged_df = pd.merge(race_df, weather_df, on="RaceID")

# Convert Rainfall values into numeric (1 for True, 0 for False)
merged_df["Rainfall"] = merged_df["Rainfall"].apply(lambda x: 1 if x == "True" else 0)

# Histogram showing distribution of finishing times
plt.figure(figsize=(8,5))
plt.hist(merged_df["Time(s)"].dropna(), bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Finishing Times")
plt.xlabel("Finishing Time (seconds)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Pie chart showing proportion of races with and without rainfall
rain_counts = merged_df["Rainfall"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(rain_counts, labels=["No Rain","Rain"], autopct="%1.1f%%", colors=["lightgreen","lightblue"])
plt.title("Proportion of Races with Rainfall")
plt.show()

# Bar chart showing average finishing time by team
team_avg = merged_df.groupby("TeamName")["Time(s)"].mean().sort_values()
plt.figure(figsize=(10,6))
plt.barh(team_avg.index, team_avg.values, color="orange")
plt.title("Average Finishing Time by Team")
plt.xlabel("Average Time (seconds)")
plt.ylabel("Team")
plt.show()

# Bar chart showing finishing positions in the first race
# Plot driver positions for the first race
first_race_id = merged_df['RaceID'].iloc[0]
first_race = merged_df[merged_df['RaceID'] == first_race_id]

plt.figure(figsize=(12, 6))
plt.bar(first_race['FullName'], first_race['Position'])

# Make driver names more visible
plt.xticks(rotation=45, ha='right', fontsize=10)  
plt.yticks(fontsize=10)  
plt.subplots_adjust(bottom=0.25)  # Add space for rotated labels

plt.xlabel("Driver", fontsize=12)
plt.ylabel("Position", fontsize=12)

# Use RaceID (or other available column) for title
title_col = 'RaceID' if 'RaceName' not in merged_df.columns else 'RaceName'
plt.title(f"Driver Positions in Race {first_race[title_col].iloc[0]}", fontsize=14)

plt.tight_layout()
plt.show()
