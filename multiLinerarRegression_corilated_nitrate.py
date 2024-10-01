# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:27:33 2024

@author: prana
"""

#MultiLinerRegression
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing 
import coreSteps1

# MultiLinearRegression







# creating feature variables 
X = coreSteps1.col2_nitrate4.drop('NITRATE', axis=1) 
y = coreSteps1.col2_nitrate4['NITRATE'] 



# creating feature variables 
#X = coreSteps.df.drop('IRON', axis=1) 
#y = coreSteps.df['IRON'] 


# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.3, random_state=101)


# creating a regression model 
model = LinearRegression() 
  
# fitting the model 
model.fit(X_train, y_train) 
  
# making predictions 
predictions = model.predict(X_test) 

#model intercept
# Printing coefficients and intercept
print(model.coef_)
print(model.intercept_)


  
# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,predictions)))


############## predictions vs actual values 

#Actual value and the predicted value
multiLinear_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': predictions})
multiLinear_diff.tail(50)


# saving the DataFrame as a CSV file 
gfg_csv_data = multiLinear_diff.to_csv('outPut/MultiLR_Scalar2.csv', index = True)


    # plot true values vs predicted values 
plt.figure(figsize=(10,10))
plt.scatter(y_test, predictions, c='crimson')
plt.yscale('log')
plt.xscale('log')


p1 = max(max(predictions), max(y_test))
p2 = min(min(predictions), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-') # plot diagonal blue line where true values = predictions
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()







residuals = y_test - predictions
plt.scatter(predictions, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()



# Writing a function to predict charges
coefficients = model.coef_
intercept = model.intercept_
def calculate_charges(PAH, BENTAZONE, NITRITE_AT_TAP,SULPHATE):
  return (PAH * coefficients[0]) + (BENTAZONE * coefficients[1]) + (NITRITE_AT_TAP * coefficients[2])+ (SULPHATE * coefficients[3]) + intercept


# Predicting charges
# Predicting charges
print('actual value is 27.2( sampleid=2020/2159) and predicted value is:  ',calculate_charges(0.01, 0.005, 0.013,14.6))


# actual value is 27.2( sampleid=2020/2159) and predicted value is 17.5 


