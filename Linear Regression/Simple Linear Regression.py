# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:25:31 2020

@author: Sonnet

"""
import matplotlib.pyplot as plt
from sklearn import *
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# Dataset
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

# Reading Data
df = pd.read_csv("FuelConsumption.csv")

# Spliting the dataset into train and test sets, 80% of the entire data for training, and the 20% for testing.
# Creating a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


# Training Data Distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='black')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Modeling Data
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

# The Coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# Outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='black')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
#plt.plot(train_x, regr.predict(train_x), '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# Evaluation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)


print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
#The higher the R-squared, the better the model fits your data. Best possible score is 1.0
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
