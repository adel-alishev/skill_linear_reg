from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('non_linear.csv', sep=',')
data.head()


def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный
    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
          source_data**n for n in range(1, degree + 1)
    ]).T

#plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')

model_ridge = Ridge(alpha=0.01)
model_linear = Ridge(alpha=0.0)
degree = 10

X = generate_degrees(data['x_train'], degree)
y = data['y_train']
# обучаем линейную регрессию с  регуляризацией
model_ridge.fit(X, y)
model_linear.fit(X, y)

x_linspace = np.linspace(data['x_train'].min(), data['x_train'].max(), num=100)

y_linspace_linear = model_linear.predict(generate_degrees(x_linspace, degree))
y_linspace_ridge = model_ridge.predict(generate_degrees(x_linspace, degree))

plt.plot(x_linspace, y_linspace_linear)
plt.plot(x_linspace, y_linspace_ridge)

plt.show()
print("Норма вектора весов Ridge \t||w|| = %.2f" % (norm(model_ridge.coef_)))
print("Норма вектора весов Linear \t||w|| = %.2f" % (norm(model_linear.coef_)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

alphas = [0.1, 0.15, 0.35 ,0.5,0.8]

best_alpha = alphas[0]
best_rmse = np.infty

for alpha in alphas:
    model_ridge = Ridge(alpha=alpha)
    # обучаем линейную регрессию с  регуляризацией
    model_ridge.fit(X_train, y_train)
    y_pred = model_ridge.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    if error < best_rmse:
        best_rmse = error
        best_alpha = alpha
    print("alpha =%.2f Ошибка %.5f" % (alpha, error))
print('\n-------\nЛучшая модель aplpha=%.2f с ошибкой RMSE=%.5f\n-------' % (best_alpha, best_rmse))