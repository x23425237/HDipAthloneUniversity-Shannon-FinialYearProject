# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:05:31 2024

@author: prana
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error ,accuracy_score
from sklearn.preprocessing import StandardScaler
import coreSteps1
import category_encoders as ce

# import Random Forest classifier

from sklearn.ensemble import RandomForestRegressor

#Drop rows with all zeros in pandas data frame
nonZero=coreSteps1.col2.loc[~(coreSteps1.col2==0).all(axis=1)]

des=nonZero.describe()
# save the statistical summary into external file
# saving the DataFrame as a CSV file 
gfg_csv_data = des.to_csv('outPut/paramSummaryNonZero.csv', index = True)




# creating feature variables 
X = nonZero.drop('NITRATE', axis=1) 
y = nonZero['NITRATE'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# check the shape of X_train and X_test
X_train.shape, X_test.shape


# instantiate the classifier 
rfr = RandomForestRegressor(random_state=0).fit(X_train, y_train)


predictions = rfr.predict(X_test)

y_pred_train =  rfr.predict(X_train)#make predictions on train set

# model evaluation on train data
print('mean_squared_error on train data : ', mean_squared_error(y_train, y_pred_train))


  
# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,predictions)))