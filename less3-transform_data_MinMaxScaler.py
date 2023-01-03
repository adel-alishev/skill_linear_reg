from sklearn.preprocessing import MinMaxScaler
import numpy as np
y = np.array([
    1.,  2.5,  2.1,  4.4,  2., 10.5,  2.,  5.,  2.,  2.],
    dtype=np.float32
)
print("Сырой датасет: %s" % y)

transformed_data = MinMaxScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)

print("Min-Max scale датасет: %s" % transformed_data)