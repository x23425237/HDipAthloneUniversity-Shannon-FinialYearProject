# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:27:33 2024

@author: prana
"""

#https://datagy.io/python-sklearn-linear-regression/
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression

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
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 



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

y_pred_train =  model.predict(X_train)#make predictions on train set

# model evaluation on train data
print('mean_squared_error on train data : ', mean_squared_error(y_train, y_pred_train))


  
# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,predictions)))




############## predictions vs actual values 

#Actual value and the predicted value
multiLinear_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': predictions})
multiLinear_diff.tail(50)


# saving the DataFrame as a CSV file 
gfg_csv_data = multiLinear_diff.to_csv('outPut/MultiLR_Scalar.csv', index = True)


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


# Get all column names excluding 'nitrate'
column_names_excluding_nitrate = [col for col in coreSteps1.col2.columns if col != 'NITRATE']

# Convert coefficients to a DataFrame
coefficients_df = pd.DataFrame({'Feature': column_names_excluding_nitrate, 'Coefficient': model.coef_})

# Save coefficients to a CSV file
coefficients_df.to_csv('outPut/MLAllcoefficients.csv', index=False)



# Visualize the coefficients in a graph
plt.figure(figsize=(10, 6))
plt.plot(model.coef_, marker='o', linestyle='')
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient value')
plt.title('Visualization of Ridge Model Coefficients')
plt.tight_layout()  # Adjust the layout to avoid overlap
plt.grid(True)
plt.show()