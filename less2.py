from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
boston_dataset = load_boston()


features = boston_dataset.data
y = boston_dataset.target
print('Матрица Объекты Х Фичи (размерность): %s %s' %features.shape)
print('\nЦелевая переменная y (размерность): %s' %y.shape)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(features, y)

print('Аналитически определенные коэффициенты: \n%s'%reg.coef_)

y_pred = reg.predict(features)
y_true = y

print('---------- Metrics ----------')
MAE = mean_absolute_error(y_pred, y_true)
RMSE = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print('Mean Absolute Error (MAE): %s' %MAE)
print('Mean Squared Error (RMSE): %s' %RMSE)
print('R2 Score: %s' %r2)