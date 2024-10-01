# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:27:33 2024

@author: prana
"""
#https://medium.com/@bernardolago/mastering-ridge-regression-a-key-to-taming-data-complexity-98b67d343087

#MultiLinerRegression
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn import preprocessing 
import coreSteps1

# MultiLinearRegression

# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 



# creating feature variables 
#X = coreSteps.df.drop('IRON', axis=1) 
#y = coreSteps.df['IRON'] 


# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.3, random_state=101)

# Instantiate the Ridge Regression model
ridge_reg = Ridge(alpha=1)  # alpha is the hyperparameter equivalent to lambda

# Train the model
ridge_reg.fit(X_train, y_train)

# Make predictions
y_pred = ridge_reg.predict(X_test)# make predictions on test set

y_pred_train = ridge_reg.predict(X_train)#make predictions on train set

# model evaluation on train data
print('mean_squared_error on train data : ', mean_squared_error(y_train, y_pred_train))
  

# model evaluation on the test data
print('mean_squared_error on test data : ', mean_squared_error(y_test, y_pred)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,y_pred)))




############## predictions vs actual values 

#Actual value and the predicted value
ridge_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})



# saving the DataFrame as a CSV file 
gfg_csv_data = ridge_diff.to_csv('outPut/ridgeReg.csv', index = True)


    # plot true values vs predicted values 
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Petal Width')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


# Get all column names excluding 'nitrate'
column_names_excluding_nitrate = [col for col in coreSteps1.col2.columns if col != 'NITRATE']

# Convert coefficients to a DataFrame
coefficients_df = pd.DataFrame({'Feature': column_names_excluding_nitrate, 'Coefficient': ridge_reg.coef_})

# Save coefficients to a CSV file
coefficients_df.to_csv('outPut/redgeRegcoefficients.csv', index=False)


# Visualize the coefficients in a graph
plt.figure(figsize=(10, 6))
plt.plot(ridge_reg.coef_, marker='o', linestyle='')
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient value')
plt.title('Visualization of Ridge Model Coefficients')
plt.tight_layout()  # Adjust the layout to avoid overlap
plt.grid(True)
plt.show()
