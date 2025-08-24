import matplotlib.pyplot as plt
import seaborn as sns
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

# Box Plot: Finishing times by team
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_df, x="TeamName", y="Time(s)")
plt.xticks(rotation=45, ha='right')
plt.title("Box Plot of Finishing Times by Team")
plt.xlabel("Team")
plt.ylabel("Time (s)")
plt.show()
# Line Graph: Finishing times across drivers in first race
plt.figure(figsize=(10, 6))
first_race = merged_df[merged_df['RaceID'] == merged_df['RaceID'].iloc[0]]
plt.plot(first_race['FullName'], first_race['Time(s)'], marker='o')
plt.xticks(rotation=45, ha='right')
plt.title("Line Graph of Finishing Times (First Race)")
plt.xlabel("Driver")
plt.ylabel("Time (s)")
plt.show()

# Scatter Plot: Finishing time vs position
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['Position'], merged_df['Time(s)'], alpha=0.7)
plt.title("Scatter Plot of Position vs Time")
plt.xlabel("Position")
plt.ylabel("Time (s)")
plt.show()

# Extract season (year) from RaceID
merged_df["Season"] = merged_df["RaceID"].str.split("_").str[0].astype(int)

# Filter only 2024 races
season_2024 = merged_df[merged_df["Season"] == 2024]

# Create pivot table for 2024 only
pivot_2024 = season_2024.pivot_table(
    index="RaceID", columns="Rainfall", values="Position", aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_2024, annot=True, cmap="coolwarm", cbar=True, fmt=".1f")
plt.title("2024 Season: Avg Finishing Position (Rain vs Non-Rain)")
plt.xlabel("Rainfall Condition")
plt.ylabel("Race ID")
plt.show()

# Bar Graph: Number of wins by team
wins_by_team = merged_df[merged_df["Position"] == 1]["TeamName"].value_counts()

plt.figure(figsize=(10, 6))
wins_by_team.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Number of Wins by Team")
plt.xlabel("Team")
plt.ylabel("Wins")
plt.xticks(rotation=45, ha='right')
plt.show()

# Histogram: Distribution of finishing times
plt.figure(figsize=(8, 6))
plt.hist(merged_df['Time(s)'].dropna(), bins=20, edgecolor='black')
plt.title("Histogram of Finishing Times")
plt.xlabel("Time (s)")
plt.ylabel("Frequency")
plt.show()
