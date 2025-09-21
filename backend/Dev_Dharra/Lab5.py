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

# Drop NaN Time(s)
df_2024 = df_2024.dropna(subset=['Time(s)'])

# Define features and target
X = df_2024[['Position']]   # independent variable
y = df_2024['Time(s)']      # dependent variable

# Linear Regression 
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

# Polynomial Regression (degree=5) 
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Create smooth curve for polynomial fit 
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_poly = poly_reg.predict(X_range_poly)

# Plot side by side 
plt.figure(figsize=(12,5))

# Linear Regression subplot
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X, y_pred_linear, color='blue', label="Linear Fit")
plt.xlabel("Position")
plt.ylabel("Time(s)")
plt.title("Linear Regression (2024 Season)")
plt.legend()

# Polynomial Regression subplot
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X_range, y_range_poly, color='green', label="Polynomial Fit (deg=5)")
plt.xlabel("Position")
plt.ylabel("Time(s)")
plt.title("Polynomial Regression (2024 Season)")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluation 
print("Linear Regression R²:", r2_score(y, y_pred_linear))
print("Polynomial Regression R²:", r2_score(y, poly_reg.predict(X_poly)))