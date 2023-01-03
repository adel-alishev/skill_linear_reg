from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
boston_dataset = load_boston()
features = boston_dataset.data
y = boston_dataset.target
print('Матрица Объекты Х Фичи (размерность): %s %s' %features.shape)
print('\nЦелевая переменная y (размерность): %s' %y.shape)
x_train, x_test, y_train, y_test = train_test_split(features, y, random_state=44)
print(x_train.shape, x_test.shape)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print(r2_score(y_test,y_pred))


