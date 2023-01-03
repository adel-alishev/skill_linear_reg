import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

x = np.linspace(1,10,num=10)
y = np.array(
    [1.,  3.,  4.,  2., 10.,  5.,  5.,  2.,  5., 10.],
    dtype=np.float32
)
y_transform = StandardScaler().fit_transform(y.reshape(-1,1))
print(y_transform)
plt.scatter(x,y_transform)
plt.show()