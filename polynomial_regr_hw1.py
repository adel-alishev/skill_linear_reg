import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('non_linear.csv')
def generate_degrees(source_data: list, degree: int):
    """Функция, которая принимает на вход одномерный массив, а возвращает n-мерный

    Для каждой степени от 1 до  degree возводим x в эту степень
    """
    return np.array([
        source_data ** n for n in range(1, degree + 1)
    ]).T

r2_result=[]
for i in range(1, 20):
    X = generate_degrees(data['x_train'], degree=i)
    reg = LinearRegression()
    reg.fit(X, data.y_train)
    y_pred = reg.predict(X)
    r2 = r2_score(data.y_train, y_pred)
    r2_result.append(r2)

print(r2_result)
print(max(r2_result), np.argmax(r2_result)+1)

