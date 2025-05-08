import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Load the dataset
df = pd.read_csv('house_prices.csv')  # Replace with your dataset path

# 2. Preprocessing
# Drop rows with missing target
df = df.dropna(subset=['SalePrice'])

# Fill missing values for features
df.fillna(df.median(numeric_only=True), inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 3. Feature and target split
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Smart Regression Model: XGBoost with hyperparameter tuning
xgb_model = XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1.0]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# 7. Evaluation
y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²: {r2:.3f}")

# 8. Plot Predictions vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.grid(True)
plt.show()
