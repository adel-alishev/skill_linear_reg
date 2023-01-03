import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
x = np.linspace(1,10, num=10).reshape(-1,1)
#данные с выбросами
y = [2,3,4,5,5,7,8,10,9,11]
plt.scatter(x,y)
plt.show()
#обучим модель измерим метрику
reg = LinearRegression().fit(x,y)
y_pred = reg.predict(x)
print('RMSE до логарифмирования: %s' %mean_squared_error(y, y_pred))
print('R2 до логарифмирования: %s' %r2_score(y, y_pred))

#преобразуем данные через логарифмирование

x = np.log(x)
plt.scatter(x,y)
plt.show()
#обучим модель измерим метрику
reg = LinearRegression().fit(x,y)
y_pred = reg.predict(x)
print('RMSE после логарифмирования: %s' %mean_squared_error(y, y_pred))
print('R2 после логарифмирования: %s' %r2_score(y, y_pred))
