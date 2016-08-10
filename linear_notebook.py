import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)
#print(data.head())
data['test'] = data['TV'] * data['Radio']
print(data.head())

"""
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16,8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
"""

"""
lm = smf.ols(formula='Sales ~ TV', data=data).fit()
#print(lm.params)


X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
#print(X_new.head())

preds = lm.predict(X_new)

data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, preds, c='red')
plt.xlim([-5,350])
plt.show()
"""

"""
Multivariate Linear
lm = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
print(lm.params)
print(lm.summary())
"""


"""
# Regression using Sci-kit learn
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X,y)

print(lm.intercept_)
print(lm.coef_)
zip(feature_cols, lm.coef_)
print(lm.score(X, y))
"""

# Categorical Predictors

np.random.seed(12345)

nums = np.random.rand(len(data))
mask_large = nums > 0.5

data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'

data['IsLarge'] = data.Size.map({'small':0, 'large':1})

"""
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge']
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X, y)
zip(feature_cols, lm.coef_)
print(lm.coef_)
"""
"""
# More than one category
np.random.seed(123456)
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
print(data.head())

area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]

data = pd.concat([data, area_dummies], axis=1)
print(data.head())

feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X,y)

zip(feature_cols, lm.coef_)

print(lm.coef_)

lm_OLS = smf.ols(formula='Sales ~ TV + Radio + IsLarge + Area_suburban + Area_urban', data=data).fit()
print(lm_OLS.summary())
"""