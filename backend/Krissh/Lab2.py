import pandas as pd
import numpy as np
import sqlite3

# Connecting and extracting data from SQLite databases
conn = sqlite3.connect("data/race_data.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

conn = sqlite3.connect("data/weather_data.db")
weather_df = pd.read_sql("SELECT * FROM weather_table", conn)
conn.close()

# Displaying first few rows of both datasets
print("Race Data Sample:")
print(race_df.head())

print("\nWeather Data Sample:")
print(weather_df.head())

# Showing dataset information and structure
print("\nRace Data Info:")
print(race_df.info())

print("\nWeather Data Info:")
print(weather_df.info())

# Checking dataset shapes
print("\nRace Data Shape:", race_df.shape)
print("Weather Data Shape:", weather_df.shape)

# Generating summary statistics
print("\nRace Data Summary:")
print(race_df.describe(include="all"))

print("\nWeather Data Summary:")
print(weather_df.describe(include="all"))

# Checking for missing values
print("\nMissing Values in Race Data:")
print(race_df.isnull().sum())

print("\nMissing Values in Weather Data:")
print(weather_df.isnull().sum())

# Removing duplicate rows
race_df = race_df.drop_duplicates()
weather_df = weather_df.drop_duplicates()

# Merging race and weather data using RaceID and RaceName
merged_df = pd.merge(race_df, weather_df, on=["RaceID", "RaceName"], how="inner")
print("\nMerged Data Sample:")
print(merged_df.head())

# Converting race finishing times to NumPy array and performing basic statistics
time_array = race_df["Time(s)"].to_numpy()

times_numeric = pd.to_numeric((time_array), errors="coerce")

# Step 2: Drop NaN values if any
times_numeric = times_numeric[~np.isnan(times_numeric)]

# Step 3: Compute statistics
mean_time = np.mean(times_numeric)
fastest_time = np.min(times_numeric)
slowest_time = np.max(times_numeric)
std_dev = np.std(times_numeric)

print("Clean Times:", times_numeric[:10])
print("Mean Time:", mean_time)
print("Fastest Time:", fastest_time)
print("Slowest Time:", slowest_time)
print("Standard Deviation:", std_dev)

# Converting rainfall values to NumPy array and performing basic statistics
rain_array = weather_df["Rainfall"].to_numpy()

rain_bool = np.where(rain_array == 'True', 1, 0)

mean_rain = np.mean(rain_bool)   # proportion of rainy races
total_rain = np.sum(rain_bool)   # total rainy races
no_rain = len(rain_bool) - total_rain  # total dry races

print("\nRainfall Boolean Array:", rain_bool)
print("Mean Rainfall (Proportion of Rainy Races):", mean_rain)
print("Total Races with Rain:", total_rain)
print("Total Races without Rain:", no_rain)