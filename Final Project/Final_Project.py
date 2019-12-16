import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

df = pd.read_csv('winequality-red.csv')

# Correlation Plot
# Based on this plot, I predit that residual sugar, free sulfur dioxide, and 
# pH do not relate to the wine quality
corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()
 
# Spearman test: to figure out which covariate will affect the wine quality
# Is my prediction correct?
# The Spearman test shows wine quality does not affect by pH and residual sugar
# when aplha = 0.05 is applied
varlist = []
notvarlist = []
for var in df.head(1):
    if var != 'quality':
        r, p = ss.spearmanr(df[var], df['quality'])
        if p < 0.05:
            varlist.append(var)
        else:
            notvarlist.append(var)
print(notvarlist)

# Multi linear regression
X = df[varlist]
Y = df['quality']
regr = LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)