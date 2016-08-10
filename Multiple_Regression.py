import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

"""
# Another way to do categorical data using pd.Categorical(df.column).labels
df = pd.read_csv('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data',index_col=0)

X = df.copy()
y = X.pop('chd')

#print(y.groupby(X.famhist).mean())

df['famhist_ord'] = pd.Categorical(df.famhist).labels
print(df.head())

est = smf.ols(formula="chd ~ famhist_ord", data=df).fit()
print(est.summary())
"""
"""
df = pd.read_csv('https://raw.githubusercontent.com/statsmodels/statsmodels/master/statsmodels/datasets/randhie/src/randhie.csv')

df["logincome"] = np.log1p(df.income)
print(df[['mdvis','logincome','hlthp']].tail())



df.plot(kind='scatter', x='logincome', y='mdvis', alpha=0.3)
plt.xlabel('Log Income')
plt.ylabel('Number of Visits')

income_linspace = np.linspace(df.logincome.min(), df.logincome.max(), 100)
"""
"""
# Parallel lines
est = smf.ols(formula='mdvis ~ logincome + hlthp', data=df).fit()


plt.plot(income_linspace, est.params[0] + est.params[1]*income_linspace + est.params[2]*0,'r')
plt.plot(income_linspace, est.params[0] + est.params[1]*income_linspace + est.params[2]*1,'g')
plt.show()
"""
"""
# Interactions

est = smf.ols(formula='mdvis ~ hlthp * logincome', data=df).fit()
plt.plot(income_linspace, est.params[0] + est.params[1]*0 + est.params[2]*income_linspace +
         est.params[3]*0*income_linspace, 'r')

plt.plot(income_linspace, est.params[0] + est.params[1]*1 + est.params[2]*income_linspace +
         est.params[3]*1*income_linspace, 'g')
plt.show()
print(est.params)
"""
"""
# Polynomial regressions
df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv')
print(df.head())



plt.figure(figsize=(6*1.618,6))
plt.scatter(df.lstat, df.medv, s=10, alpha=0.3)
plt.xlabel('lstat')
plt.ylabel('medv')

x = pd.DataFrame({'lstat':np.linspace(df.lstat.min(), df.lstat.max(), 100)})

# First order
poly_1 = smf.ols(formula='medv ~ 1 + lstat', data=df).fit()
plt.plot(x.lstat, poly_1.predict(x), 'b-', label='Poly n=1 $R^2$=%.2f' % poly_1.rsquared, alpha=0.9)


# Second order
poly_2 = smf.ols(formula='medv ~ 1 + lstat + I(lstat ** 2)', data = df).fit()
plt.plot(x.lstat, poly_2.predict(x), 'g-', label='Poly n=2 $R^2$=%.2f' % poly_2.rsquared, alpha=0.9)


# Third order
poly_3 = smf.ols(formula='medv ~ 1 + lstat + I(lstat ** 2) + I(lstat ** 3)', data = df).fit()
plt.plot(x.lstat, poly_3.predict(x), 'r-', label='Poly n=3 $R^2$=%.2f' % poly_3.rsquared, alpha=0.9)


plt.legend()
plt.show()

print(poly_2.summary())
print(poly_3.summary())
"""