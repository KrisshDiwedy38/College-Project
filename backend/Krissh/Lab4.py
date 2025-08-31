import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sqlite3

# Load race data from SQLite database
conn = sqlite3.connect("data/race_data.db")
race_df = pd.read_sql("SELECT * FROM race_table", conn)
conn.close()

# Load weather data from SQLite database
conn = sqlite3.connect("data/weather_data.db")
weather_df = pd.read_sql("SELECT * FROM weather_table", conn)
conn.close()

# Data Preprocessing
print("Starting data preprocessing...")

# Check missing values
print("\nMissing values in race data:")
print(race_df.isnull().sum())
print("\nMissing values in weather data:")
print(weather_df.isnull().sum())

# Handle missing Position values 
max_position = race_df['Position'].max()
race_df['Position'].fillna(max_position + 1, inplace=True)

# Handle missing Time values (483 missing)
# Create indicator for time availability
race_df['Time_Available'] = ~race_df['Time(s)'].isnull()

# For finished drivers with missing times, estimate using interpolation
for race_id in race_df['RaceID'].unique():
    race_mask = race_df['RaceID'] == race_id
    finished_mask = race_df['Status'] == 'Finished'
    
    race_data = race_df[race_mask & finished_mask].copy()
    
    if race_data['Time(s)'].isnull().sum() > 0 and race_data['Time(s)'].notna().sum() > 1:
        race_data_sorted = race_data.sort_values('Position')
        race_data_sorted['Time(s)'] = race_data_sorted['Time(s)'].interpolate(method='linear')
        race_df.loc[race_mask & finished_mask, 'Time(s)'] = race_data_sorted['Time(s)'].values

# Keep NaN for DNF drivers 
dnf_mask = race_df['Status'] != 'Finished'
race_df.loc[dnf_mask, 'Time(s)'] = np.nan

# Create useful features
race_df['Race_Year'] = race_df['RaceID'].str.extract(r'(\d{4})').astype(int)
race_df['Race_Number'] = race_df['RaceID'].str.extract(r'_(\d+)').astype(int)
race_df['Finished'] = (race_df['Status'] == 'Finished').astype(int)
race_df['Points_Finish'] = (race_df['Position'] <= 10).astype(int)  # Top 10 get points
race_df['Podium_Finish'] = (race_df['Position'] <= 3).astype(int)   # Top 3 get podium

# Process weather data
weather_df['Weather_Condition'] = weather_df['Rainfall'].map({True: 'Wet', False: 'Dry'})
weather_df['Race_Year'] = weather_df['RaceID'].str.extract(r'(\d{4})').astype(int)
weather_df['Race_Number'] = weather_df['RaceID'].str.extract(r'_(\d+)').astype(int)

# Merge race and weather data
merged_df = race_df.merge(weather_df, on='RaceID', how='left', suffixes=('', '_weather'))

# Handle any missing weather data
merged_df['Rainfall'].fillna(False, inplace=True)
merged_df['Weather_Condition'].fillna('Dry', inplace=True)

# Remove duplicate columns from merge
duplicate_cols = [col for col in merged_df.columns if col.endswith('_weather')]
merged_df.drop(columns=duplicate_cols, inplace=True)

# Remove duplicates and validate
merged_df = merged_df.drop_duplicates(subset=['RaceID', 'DriverCode'])
merged_df['Position'] = merged_df['Position'].abs()

print("\nPreprocessing complete!")
print(f"Final dataset shape: {merged_df.shape}")
print(f"\nRemaining missing values:")
print(merged_df.isnull().sum()[merged_df.isnull().sum() > 0])

# Display sample of processed data
print("\nSample of processed data:")
print(merged_df.head(10))