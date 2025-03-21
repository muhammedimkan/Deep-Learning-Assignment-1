# 4. Given a csv file for predicting the house prices. Split the dataset as train and validation set in the ratio 80:20. 
#    a) Train a linear regression model. Compute the MSE loss on validation data. Print the feature coefficient names and their values.
#    b) Train the model applying L1 (Lasso) Regularization. Compute the MSE loss on validation data. Print the feature coefficient names and their values.
#    c) Train the model applying L2 (Ridge) Regularization. Compute the MSE loss on validation data. Print the feature coefficient names and their values.

#    Explain and justify the feature coefficient values you observe for each model.
#    csv file source: https://github.com/selva86/datasets/blob/master/BostonHousing.csv


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the BostonHousing dataset from GitHub using the Python engine
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv?raw=true"
df = pd.read_csv(url, engine='python')

# Display the first few rows of the dataset (optional)
print("Dataset Head:")
print(df.head())

# In the BostonHousing dataset, the target variable is 'medv' (median house value)
# and the features are all the other columns.
target_col = 'medv'
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].values
y = df[target_col].values

# Split the dataset into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# Function to train a model and print evaluation metrics
def train_and_evaluate(model, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    
    # Print the results
    print(f"\n{model_name} Results:")
    print(f"MSE on validation set: {mse:.2f}")
    
    # Retrieve and print feature coefficients
    coef = model.coef_
    for feature, value in zip(feature_cols, coef):
        print(f"Feature: {feature:10s} Coefficient: {value:.4f}")
    
    return mse, coef

# a) Standard Linear Regression (OLS)
lr_model = LinearRegression()
mse_lr, coef_lr = train_and_evaluate(lr_model, "Linear Regression (OLS)")

# b) Lasso Regression (L1 Regularization)
lasso_model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
mse_lasso, coef_lasso = train_and_evaluate(lasso_model, "Lasso Regression (L1)")

# c) Ridge Regression (L2 Regularization)
ridge_model = Ridge(alpha=0.1, random_state=42)
mse_ridge, coef_ridge = train_and_evaluate(ridge_model, "Ridge Regression (L2)")

# Plot MSE values for comparison
models = ['OLS', 'Lasso', 'Ridge']
mse_values = [mse_lr, mse_lasso, mse_ridge]

plt.figure(figsize=(8, 5))
plt.bar(models, mse_values, color=['blue', 'green', 'red'])
plt.ylabel("MSE on Validation Set")
plt.title("MSE Comparison Across Models")
plt.grid(True)
plt.show()

# Plot the coefficient values for each model for visual comparison
x_axis = np.arange(len(feature_cols))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x_axis - width, coef_lr, width, label='OLS')
plt.bar(x_axis, coef_lasso, width, label='Lasso')
plt.bar(x_axis + width, coef_ridge, width, label='Ridge')
plt.xticks(x_axis, feature_cols, rotation=45)
plt.ylabel("Coefficient Value")
plt.title("Feature Coefficients Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# RESULT

# Dataset Head:

#       crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio       b  lstat  medv
# 0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3  396.90   4.98  24.0
# 1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8  396.90   9.14  21.6
# 2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8  392.83   4.03  34.7
# 3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7  394.63   2.94  33.4
# 4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7  396.90   5.33  36.2

# 1. Linear Regression (OLS) Results:

#     MSE on validation set: 24.29

#     Feature: crim       Coefficient: -0.1131
#     Feature: zn         Coefficient: 0.0301
#     Feature: indus      Coefficient: 0.0404
#     Feature: chas       Coefficient: 2.7844
#     Feature: nox        Coefficient: -17.2026
#     Feature: rm         Coefficient: 4.4388
#     Feature: age        Coefficient: -0.0063
#     Feature: dis        Coefficient: -1.4479
#     Feature: rad        Coefficient: 0.2624
#     Feature: tax        Coefficient: -0.0106
#     Feature: ptratio    Coefficient: -0.9155
#     Feature: b          Coefficient: 0.0124
#     Feature: lstat      Coefficient: -0.5086

# 2. Lasso Regression (L1) Results:

#     MSE on validation set: 25.16

#     Feature: crim       Coefficient: -0.1042
#     Feature: zn         Coefficient: 0.0349
#     Feature: indus      Coefficient: -0.0168
#     Feature: chas       Coefficient: 0.9200
#     Feature: nox        Coefficient: -0.0000
#     Feature: rm         Coefficient: 4.3117
#     Feature: age        Coefficient: -0.0151
#     Feature: dis        Coefficient: -1.1515
#     Feature: rad        Coefficient: 0.2392
#     Feature: tax        Coefficient: -0.0130
#     Feature: ptratio    Coefficient: -0.7322
#     Feature: b          Coefficient: 0.0131
#     Feature: lstat      Coefficient: -0.5647

# 3. Ridge Regression (L2) Results:

#     MSE on validation set: 24.30

#     Feature: crim       Coefficient: -0.1124
#     Feature: zn         Coefficient: 0.0305
#     Feature: indus      Coefficient: 0.0349
#     Feature: chas       Coefficient: 2.7503
#     Feature: nox        Coefficient: -15.9245
#     Feature: rm         Coefficient: 4.4458
#     Feature: age        Coefficient: -0.0073
#     Feature: dis        Coefficient: -1.4296
#     Feature: rad        Coefficient: 0.2600
#     Feature: tax        Coefficient: -0.0108
#     Feature: ptratio    Coefficient: -0.9008
#     Feature: b          Coefficient: 0.0124
#     Feature: lstat      Coefficient: -0.5109