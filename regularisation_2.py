import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance

data = pd.read_csv("non_linear.csv")
data = data [(data.x_train >1) & (data.x_train<5)].copy()

X = data['x_train'].values.reshape(-1,1)
y = data['y_train'].values

plt.scatter(data.x_train, data.y_train, 40, 'g', 'o', alpha=0.8)
plt.show()

def gradient(X, y, w) -> np.array:

    #кол-во примеров обучающей выборки
    n = X.shape[0]
    y_hat = X.dot(w.T)
    #вычисляем ошибку
    error = y - y_hat
    #градиент функции
    grad = np.multiply(X, error).sum(axis=0)*(-1.0)*2.0/n
    return grad, error

def eval_w_next(X,y, eta, w_current):
    #вычислить градиент
    grad, error = gradient(X, y, w_current)
    #шаг градиентного спуска
    w_next = w_current - eta*grad
    #условие сходимости

    weight_evolution = np.sqrt(((w_next-w_current)**2).sum(axis=1))
    return (w_next, weight_evolution, grad)

def gradient_descent(X, y, eta=0.01, epsilon=0.001):
    m = X.shape[1]#количество фичей - размерность у градиента
    w = np.random.random(m).reshape(1,-1)
    w_next, weight_evolution, grad = eval_w_next(X, y, eta, w)
    step = 0
    while weight_evolution > epsilon:
        w = w_next
        w_next, weight_evolution, grad = eval_w_next(X, y, eta, w)
        step +=1
        if step %100 ==0:
            print("step %s, |w-w_next| = %0.5f, rrad = %s" % (step, weight_evolution, grad))
    return w

X = data.x_train.values.reshape(-1,1)
n = X.shape[0]
X = np.hstack([np.ones(n).reshape(-1,1),
               X])
print(X.shape)
w = gradient_descent(X, data.y_train.values.reshape(-1,1), eta=0.008, epsilon=0.0001)

