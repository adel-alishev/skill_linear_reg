import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv('non_linear.csv')
def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный

    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
        source_data ** n for n in range(1, degree + 1)
    ]).T
rmse_test  = []
rmse_train  = []
xx = []
for i in range(3,8):
    x = generate_degrees(df['x_train'], degree=i)
    y = df.y_train.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    model = Ridge(alpha=0).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    print('degree: %.3f' % i)
    print('Качество на валидации: %.3f' % mean_squared_error(y_test, y_pred))
    print('Качество на обучение: %.3f' % mean_squared_error(y_train, y_pred_train))
    xx.append(i)
    rmse_test.append(mean_squared_error(y_test, y_pred))
    rmse_train.append(mean_squared_error(y_train, y_pred_train))
    print('-------------------------------------------')

plt.plot(xx, rmse_train, 'g', label = 'rmse train')
plt.plot(xx, rmse_test, 'b', label='rmse valid')
plt.title('Regularization\nRMSE')
plt.xlabel('degree polynom')
plt.ylabel('RMSE')
plt.legend(loc='upper center')
plt.show()