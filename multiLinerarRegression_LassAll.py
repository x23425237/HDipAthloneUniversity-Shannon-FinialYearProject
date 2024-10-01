# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:27:33 2024

@author: prana
"""
# https://www.statology.org/lasso-regression-in-python/
##https://medium.com/@rshowrav/lasso-regression-in-python-923f4914e3ca

#MultiLinerRegression
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing 
import coreSteps1






# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 


# creating feature variables 
#X = coreSteps.df.drop('IRON', axis=1) 
#y = coreSteps.df['IRON'] 


# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.3, random_state=101)


#define model
lasso_model = Lasso().fit(X_train,y_train) # defalut alpha=1.0 

#model intercept
lasso_model.intercept_

# model co-efficient
lasso_model.coef_


# do the predictions on the test data 
y_pred = lasso_model.predict(X_test)

y_pred_train =  lasso_model.predict(X_train)#make predictions on train set

# model evaluation on train data
print('mean_squared_error on train data : ', mean_squared_error(y_train, y_pred_train))



# calculate the RMSE
np.sqrt(mean_squared_error(y_test,y_pred))

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, y_pred)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,y_pred)))




#Actual value and the predicted value
lasso_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})



# saving the DataFrame as a CSV file 
gfg_csv_data = lasso_diff.to_csv('outPut/lasso_All.csv', index = True)



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

# Get all column names excluding 'nitrate'
column_names_excluding_nitrate = [col for col in coreSteps1.col2.columns if col != 'NITRATE']

# Convert coefficients to a DataFrame
coefficients_df = pd.DataFrame({'Feature': column_names_excluding_nitrate, 'Coefficient': lasso_model.coef_})

# Save coefficients to a CSV file
coefficients_df.to_csv('outPut/lasso_coefficients.csv', index=False)


# Visualize the coefficients in a graph
plt.figure(figsize=(10, 6))
plt.plot(lasso_model.coef_, marker='o', linestyle='')
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient value')
plt.title('Visualization of Ridge Model Coefficients')
plt.tight_layout()  # Adjust the layout to avoid overlap
plt.grid(True)
plt.show()





############ hyperparameter tuning (and cross validation)
#https://medium.com/@rshowrav/lasso-regression-in-python-923f4914e3ca

# Set up the Lasso Regression model
lasso = Lasso()

# Set up the hyperparameter grid
param_grid = {'alpha': np.logspace(-4, 0, 50)}

# Set up the GridSearchCV object
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best alpha value and mean squared error
print("Best alpha value: ", grid_search.best_params_['alpha'])
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)


 

#define model
lasso_model = Lasso(0.002).fit(X_train,y_train)

#model intercept
lasso_model.intercept_

# model co-efficient
lasso_model.coef_


# do the predictions on the test data 
y_pred = lasso_model.predict(X_test)

y_pred_train =  lasso_model.predict(X_train)#make predictions on train set

# model evaluation on train data
print('mean_squared_error on train data : ', mean_squared_error(y_train, y_pred_train))


# calculate the RMSE
np.sqrt(mean_squared_error(y_test,y_pred))



# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, y_pred)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,y_pred)))




#Actual value and the predicted value
lasso_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})



# saving the DataFrame as a CSV file 
gfg_csv_data = lasso_diff.to_csv('outPut/lassoAlpha_All.csv', index = True)



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

# Get all column names excluding 'nitrate'
column_names_excluding_nitrate = [col for col in coreSteps1.col2.columns if col != 'NITRATE']

# Convert coefficients to a DataFrame
coefficients_df = pd.DataFrame({'Feature': column_names_excluding_nitrate, 'Coefficient': lasso_model.coef_})

# Save coefficients to a CSV file
coefficients_df.to_csv('outPut/lassoAlpha_coefficients.csv', index=False)


# Visualize the coefficients in a graph
plt.figure(figsize=(10, 6))
plt.plot(lasso_model.coef_, marker='o', linestyle='')
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient value')
plt.title('Visualization of Ridge Model Coefficients')
plt.tight_layout()  # Adjust the layout to avoid overlap
plt.grid(True)
plt.show()



