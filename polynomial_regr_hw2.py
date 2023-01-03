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
class CustomLinearReg:
    def __init__(self):
        self.w=[]

    def fit(self, X, y):
        self.w = inv((X.T).dot(X)).dot(X.T).dot(y)
        return self.w

    def predict(self,X):
        return X.dot(np.array(self.w))

reg = CustomLinearReg()
reg.fit(X,Y)
y_pred = reg.predict(X)
print(y_pred)