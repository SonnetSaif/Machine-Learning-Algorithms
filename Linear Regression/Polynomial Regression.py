# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 02:04:54 2020

@author: Sonnet
"""

# shuffle data 
# np.random.shuffle(train)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('I:/Position_Salaries.csv')

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

regr = LinearRegression()
Xtr = np.asanyarray(train[['Level']])
ytr = np.asanyarray(train[['Salary']])
regr.fit (Xtr, ytr)

Xts = np.asanyarray(test[['Level']])
yts = np.asanyarray(test[['Salary']])


poly_regr = PolynomialFeatures(degree = 3)
X_poly = poly_regr.fit_transform(Xts.reshape(-1, 1))

regr_2 = LinearRegression()
regr_2.fit(X_poly, yts)
y_pred = regr_2.predict(X_poly)


plt.scatter(Xtr, ytr, color = 'blue') 
plt.plot(Xts, y_pred, color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
plt.show() 

