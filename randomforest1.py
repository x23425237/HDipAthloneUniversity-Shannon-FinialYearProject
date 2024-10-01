# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:43:32 2024

@author: prana

#pip install --upgrade category_encoders
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


# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 

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







#Actual value and the predicted value
randomforest_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': predictions})

# saving the DataFrame as a CSV file 
gfg_csv_data = randomforest_model_diff.to_csv('outPut/randomForest.csv', index = True)


    # plot true values vs predicted values 
plt.figure(figsize=(10,10))
plt.scatter(y_test, predictions, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predictions), max(y_test))
p2 = min(min(predictions), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


residuals = y_test - predictions
plt.scatter(predictions, residuals)
plt.xlabel('Predicted Petal Width')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

















