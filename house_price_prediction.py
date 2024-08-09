import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


np.random.seed(0)
n_samples = 1000

square_footage = np.random.randint(1000, 5000, size=n_samples)
bedrooms = np.random.randint(1, 6, size=n_samples)
bathrooms = np.random.randint(1, 5, size=n_samples)

prices = 100 * square_footage + 5000 * bedrooms + 3000 * bathrooms + np.random.normal(0, 10000, size=n_samples)

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})


data.to_csv('house_data.csv', index=False)
print("Dataset saved as house_data.csv")


df = pd.read_csv('house_data.csv')


X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()