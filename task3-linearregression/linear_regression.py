import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if not os.path.exists("outputs"):
    os.makedirs("outputs")


df = pd.read_csv("data/Housing.csv")
print("Dataset shape:", df.shape)
print(df.head())


print("\nMissing values per column:\n", df.isnull().sum())


numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

if 'price' in numeric_cols:
    numeric_cols.remove('price')

cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", cat_cols)


X = df[numeric_cols + cat_cols]
y = df['price']


X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("After encoding, feature shape:", X.shape)


if 'area' not in df.columns:
    # If the dataset has a slightly different name, adjust accordingly
    raise KeyError("Column 'area' not found in dataset.")

X_simple = df[['area']]
y_simple = df['price']


X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)
y_pred_s = lr_simple.predict(X_test_s)


print("\n=== Simple Regression (price ~ area) ===")
print("Intercept:", lr_simple.intercept_)
print("Coefficient (slope):", lr_simple.coef_[0])
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R²:", r2_score(y_test_s, y_pred_s))

plt.figure(figsize=(6,4))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Price vs Area")
plt.legend()
plt.savefig("outputs/simple_regression_line.png")
plt.close()


X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)
y_pred_m = lr_multi.predict(X_test_m)


print("\n=== Multiple Regression (using all features) ===")
print("Intercept:", lr_multi.intercept_)
coefs = pd.Series(lr_multi.coef_, index=X.columns)
print("Top 10 coefficients:\n", coefs.sort_values(key=abs, ascending=False).head(10))
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R²:", r2_score(y_test_m, y_pred_m))


plt.figure(figsize=(6,6))
plt.scatter(y_test_m, y_pred_m, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Multiple Regression)")
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()