import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
x = np.linspace(1,10, num=10).reshape(-1,1)
y = np.array([
    1.,  2.5,  2.1,  4.4,  2., 10.5,  2.,  5.,  2.,  2.],
    dtype=np.float32
)

print("Сырой датасет: %s" % y)

transformed_data = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
print("z-transform датасет: %s" % transformed_data)
plt.scatter(x,y,40,'g','o',alpha=0.8)
plt.show()
plt.scatter(x,transformed_data,40,'g','o',alpha=0.8)
plt.show()