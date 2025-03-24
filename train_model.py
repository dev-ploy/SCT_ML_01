#step 1:import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Fixed import
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

#step 2: Load the dataset
df = pd.read_csv("train.csv")

#step 3:handle missing data
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "SalePrice"]

df = df[features]
df = df.dropna()

#step 4:split the dataset
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

#step 5:train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#step 6:train the model
model = LinearRegression()  # LinearRegression doesn't have random_state parameter
model.fit(X_train, y_train)

#step 7:evaluate the model performance
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Model Performance:")
print(f"Mean Squared Error: ${mse:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Optional: Print coefficients to understand feature importance
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:.2f}")
print(f"Intercept: ${model.intercept_:.2f}")

# Optional: Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('prediction_results.png')
plt.close()


#step 8:save the model
joblib.dump(model,"house_price_model.pkl")