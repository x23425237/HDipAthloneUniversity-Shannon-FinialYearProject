# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:27:33 2024

@author: prana
"""
# https://www.statology.org/lasso-regression-in-python/

#MultiLinerRegression
import pandas as pd
from numpy import arange
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing 
import coreSteps1




# creating feature variables 
X = coreSteps1.col2_nitrate4.drop('NITRATE', axis=1) 
y = coreSteps1.col2_nitrate4['NITRATE'] 



# creating feature variables 
#X = coreSteps.df.drop('IRON', axis=1) 
#y = coreSteps.df['IRON'] 


# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.3, random_state=101)

#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#define model
lasso_model = Lasso().fit(X_train,y_train)

#model intercept
lasso_model.intercept_

# model co-efficient
lasso_model.coef_


# do the predictions on the test data 
y_pred = lasso_model.predict(X_test)

# calculate the RMSE
np.sqrt(mean_squared_error(y_test,y_pred))

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, y_pred)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,y_pred)))




#Actual value and the predicted value
lasso_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})



# saving the DataFrame as a CSV file 
gfg_csv_data = lasso_diff.to_csv('outPut/lasso_diff.csv', index = True)



###### residual plots and plot actual values vs predicted values 
    # plot true values vs predicted values 
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')


p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-') # plot diagonal blue line where true values = predictions
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()






residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()



# Writing a function to predict charges
coefficients = lasso_model.coef_
intercept = lasso_model.intercept_
def calculate_charges(PAH, BENTAZONE, NITRITE_AT_TAP,SULPHATE):
  return (PAH * coefficients[0]) + (BENTAZONE * coefficients[1]) + (NITRITE_AT_TAP * coefficients[2])+ (SULPHATE * coefficients[3]) + intercept


# Predicting charges
print('actual value is 27.2( sampleid=2020/2159) and predicted value is:  ',calculate_charges(0.01, 0.005, 0.013,14.6))


# actual value is 27.2( sampleid=2020/2159) and predicted value is 12.93








 


