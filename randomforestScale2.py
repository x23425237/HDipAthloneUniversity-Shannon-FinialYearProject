# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:43:32 2024

@author: prana

#pip install --upgrade category_encoders
"""
#1 Importing the libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import coreSteps1
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

# import Random Forest classifier

from sklearn.ensemble import RandomForestRegressor


#2 Importing the dataset 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# check the shape of X_train and X_test
X_train.shape, X_test.shape



# scale the data Fit and transform 
scaler = StandardScaler() # fit to train data 
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)# transform test data 
X_train = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test = pd.DataFrame(X_test_norm, columns=X_test.columns)





# dfm = X_train.melt(var_name='columns')
# g = sns.FacetGrid(dfm, col='columns')
# g = (g.map(sns.distplot, 'value'))


# instantiate the classifier 
rfr = RandomForestRegressor(random_state=0).fit(X_train, y_train)

prediction = rfr.predict(X_test)



y_pred_train =  rfr.predict(X_train)#make predictions on train set
# model evaluation on train data
print('mean_squared_error on train data : ', mean_squared_error(y_train, y_pred_train))


  
# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, prediction)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, prediction)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,prediction)))


#Actual value and the predicted value
randomforest_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': prediction})

# saving the DataFrame as a CSV file 
gfg_csv_data = randomforest_model_diff.to_csv('outPut/randomForestScale.csv', index = True)


    # plot true values vs predicted values 
plt.figure(figsize=(10,10))
plt.scatter(y_test, prediction, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(prediction), max(y_test))
p2 = min(min(prediction), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


residuals = y_test - prediction
plt.scatter(prediction, residuals)
plt.xlabel('Predicted Petal Width')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()




############### min max scalar
#https://stackabuse.com/feature-scaling-data-with-scikit-learn-for-machine-learning-in-python/

