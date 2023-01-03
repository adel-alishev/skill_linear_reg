import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x_hw = [50,60,70,100]
y_hw = [10,15,40,45]
x_hw = np.reshape(x_hw, newshape=(-1,1))
y_hw = np.array(y_hw)
print(x_hw.shape)
print(y_hw.shape)
plt.scatter(x_hw, y_hw, 40, 'g', 'o', alpha=0.8)

#print(X)
reg = LinearRegression()
reg.fit(x_hw,y_hw)
w_1 = reg.coef_
w_2 = reg.intercept_
print(w_1, w_2)

X_range = np.linspace(min(x_hw), max(x_hw), num=100)
Y_m = w_2+w_1*X_range

plt.plot(X_range, Y_m)
plt.show()