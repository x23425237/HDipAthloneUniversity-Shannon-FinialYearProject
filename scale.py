# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:45:02 2024

@author: prana
"""
#https://medium.com/towards-data-science/how-and-why-to-standardize-your-data-996926c2c832

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import coreSteps1
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor

# import Random Forest classifier

#https://learnpython.com/blog/how-to-summarize-data-in-python/
des=coreSteps1.col2.describe()
# save the statistical summary into external file
# saving the DataFrame as a CSV file 
gfg_csv_data = des.to_csv('outPut/paramSummary.csv', index = True)


# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)






# scale the data 
scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test = pd.DataFrame(X_test_norm, columns=X_test.columns)

# check the shape of X_train and X_test
X_train.shape, X_test.shape


################### non scale 

# creating feature variables 
XnonScale = coreSteps1.col2.drop('NITRATE', axis=1) 
ynonScale = coreSteps1.df['NITRATE'] 

# Split the data into training and testing sets
X_trainnonScale, X_testnonScale, y_trainnonScale, y_testnonScale = train_test_split(XnonScale, ynonScale, test_size=0.3, random_state=42)


# check the shape of X_train and X_test
X_trainnonScale.shape, X_testnonScale.shape



###########################################
#min max scaling 
#https://stackabuse.com/feature-scaling-data-with-scikit-learn-for-machine-learning-in-python/

#https://towardsdatascience.com/clearly-explained-what-why-and-how-of-feature-scaling-normalization-standardization-e9207042d971

######### ramdomForest
# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y =coreSteps1.col2['NITRATE'] 


# compute required values
scaler = MinMaxScaler() #  #  StandardScaler()
model = scaler.fit(X)
scaled_data = model.transform(X)
 
# # print scaled data
# print(scaled_data)


df = pd.DataFrame(scaled_data, columns=X.columns)


# creating feature variables 
X = df 
y =coreSteps1.col2['NITRATE'] 


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
gfg_csv_data = randomforest_model_diff.to_csv('outPut/randomForestScaleAll.csv', index = True)


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



####################### min max scaling 


















