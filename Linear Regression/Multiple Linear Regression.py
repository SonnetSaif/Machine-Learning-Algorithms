# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 00:26:33 2020

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


# Modeling Data
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)


# The coefficients
print('Coefficients: ', regr.coef_)
print("Intercepts: ", regr.intercept_)


# Prediction 
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))


# Explained Variance Regression Score: The best possible score is 1.0, lower values are worse
print('Variance score: %.2f' % regr.score(x, y))
