import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.linspace(-10, 10, num=100)

plt.plot(x, x, 'b')
plt.plot(x, x ** 2, 'r')
plt.plot(x, 0.4 * x ** 3 + 2.8 * x ** 2 - 4 * x + 6, 'g')
plt.show()
data = pd.read_csv('non_linear.csv')
plt.scatter(data.x_train, data.y_train, 40, 'g', 'o')

# reg = LinearRegression()
# reg.fit(data[['x_train']], data.y_train)
# y_pred = reg.predict(data[['x_train']])
# plt.plot(data[['x_train']], y_pred, 'g')
# plt.show()


def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный

    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
        source_data ** n for n in range(1, degree + 1)
    ]).T


X = generate_degrees(data['x_train'], degree=3)
reg1 = LinearRegression()
reg1.fit(X, data.y_train)
y_pred = reg1.predict(X)
plt.scatter(data.x_train, data.y_train, 40, 'g', 'o')
plt.plot(data.x_train, y_pred)

plt.show()
