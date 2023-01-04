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

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

sgd_regressor = SGDRegressor(learning_rate='constant', eta0=0.009, fit_intercept=True, random_state=42)
w_current, epsilon = np.random.random(2), 0.001
print(w_current)
weight_evolution = []
rmse_evolution = []
for step in range(800):
    sgd_regressor = sgd_regressor.partial_fit(X_train, y_train)
    weight_evolution.append(
        distance.euclidean(w_current, sgd_regressor.coef_)
    )
    if weight_evolution[-1]<epsilon:
        print("Итерации остановлены на шаге %d" % step)
        break
    rmse_evolution.append(
        mean_squared_error(y_valid, sgd_regressor.predict(X_valid))
    )
    w_current = sgd_regressor.coef_.copy()
plt.scatter(range(step), rmse_evolution)
plt.show()
