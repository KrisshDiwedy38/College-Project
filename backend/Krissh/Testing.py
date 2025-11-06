
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .Lab4 import preprocessing

merged_df = preprocessing()

df_2024 = merged_df[merged_df['Race_Year'] == 2024].copy()

# Select one driver - let's find who has the most races in 2024
driver_counts = df_2024['DriverCode'].value_counts()
print("Available drivers and their race counts in 2024:")
print(driver_counts.head(10))
print()

# Select the driver with most races (or you can manually specify)
selected_driver = driver_counts.index[0]  # Driver with most races
print(f"Selected driver: {selected_driver}")

# Filter for selected driver only
driver_df = df_2024[df_2024['DriverCode'] == selected_driver].copy()
driver_name = driver_df['FullName'].iloc[0] if not driver_df.empty else selected_driver

# Drop NaN Time(s) - only analyze races where driver finished with a time
driver_df = driver_df.dropna(subset=['Time(s)']).copy()

print(f"Analyzing {driver_name} ({selected_driver}) - {len(driver_df)} races with finish times in 2024")
print()

if len(driver_df) < 3:
    print("Not enough data points for meaningful regression analysis")
else:
    # Analysis 1: Race Number vs Time (season progression)
    print("=== ANALYSIS 1: TIME PROGRESSION (Race Number vs Finish Time) ===")
    X1 = driver_df[['Race_Number']]
    y1 = driver_df['Time(s)']
    
    # Linear Regression 
    lin_reg1 = LinearRegression()
    lin_reg1.fit(X1, y1)
    y_pred_linear1 = lin_reg1.predict(X1)
    
    # Polynomial Regression (degree=2 for single driver)
    poly1 = PolynomialFeatures(degree=2)
    X_poly1 = poly1.fit_transform(X1)
    poly_reg1 = LinearRegression()
    poly_reg1.fit(X_poly1, y1)
    y_pred_poly1 = poly_reg1.predict(X_poly1)
    
    # Create smooth curve
    X_range1 = np.linspace(X1.min(), X1.max(), 100).reshape(-1, 1)
    X_range_poly1 = poly1.transform(X_range1)
    y_range_poly1 = poly_reg1.predict(X_range_poly1)
    
    plt.figure(figsize=(15,5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X1, y1, color='red', s=80, alpha=0.8, label="Actual Times")
    plt.plot(X1, y_pred_linear1, color='blue', linewidth=2, label="Linear Trend")
    plt.plot(X_range1, y_range_poly1, color='green', linewidth=2, label="Polynomial Trend")
    plt.xlabel("Race Number")
    plt.ylabel("Finish Time (seconds)")
    plt.title(f"{driver_name}\nRace Number vs Finish Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()