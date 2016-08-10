import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

n_samples, n_features = 50, 2
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)

clf = linear_model.SGDRegressor()
clf.fit(X, y)

regr = linear_model.LinearRegression()

regr.fit(X, y)

print('SGD Coefficients: \n', clf.coef_)
print('LS Coefficients: \n', regr.coef_)

print('SGD Residual sum of squares : %.2f' % np.mean((clf.predict(X) - y)**2))
print('Residual sum of squares : %.2f' % np.mean((regr.predict(X) - y)**2))

print(X[:,0:1])

plt.scatter(X[:,0],y, color='black')
plt.scatter(X[:,1],y, color='red')
plt.plot(X, regr.predict(X), color='orange', label='SGD')
plt.plot(X, clf.predict(X), color='blue', label='OLS Method')
plt.legend()
plt.show()