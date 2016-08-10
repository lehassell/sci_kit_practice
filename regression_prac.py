import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Target is the integer index of the category
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

# Get Basics out
print('Coefficients: \n', regr.coef_)

# MSE
print('Residual sum of squares : %.2f' % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2))

# Explained variance (1 is perfect)
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot it all

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='orange')
plt.xticks()
plt.yticks()
plt.show()
