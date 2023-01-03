import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
X = np.array([[1,50],
                [2,70],
                [3,80],
                [4,110],
                [5,90]])
Y = np.array([[40],
              [50],
              [50],
              [70],
              [60]])
# w = ((((X^T)*X)^-1)*X^T)*Y
w = inv(
    (X.T).dot(X)
).dot(X.T
      ).dot(Y)
w_1 = w[0]
w_2 = w[1]
#y = w_1 + w_2 * x
##график
#границы графика
margin = 10
X_min = X[:,1].min()-margin
X_max = X[:,1].max()+margin

X_range = np.linspace(X_min, X_max, num=100)
Y_m = w_1+w_2*X_range
#исходные точки
plt.scatter(X[:,1], Y[:,0], s=50, c='r', marker='o',
            alpha=0.8)
# пресказанная модель
plt.plot(X_range, Y_m)
plt.show()