import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def f(x):
    return x * np.sin(x)

x_plot = np.linspace(0, 10, 100)

x = np.linspace(0,10, 100)

rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.plot(x_plot, f(x_plot), label='ground truth')

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label='degree %d' % degree)


plt.legend(loc='lower left')
plt.show()

plt.plot(x_plot, f(x_plot), label='ground truth')

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label='degree %d' % degree)


plt.legend(loc='lower left')
plt.show()