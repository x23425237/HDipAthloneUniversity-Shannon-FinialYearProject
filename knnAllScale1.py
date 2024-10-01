# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:51:40 2024

@author: prana
"""

#KNN
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





# creating feature variables 
X = coreSteps1.col2.drop('NITRATE', axis=1) 
y = coreSteps1.df['NITRATE'] 



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
# Scale the features using StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Initialize the KNN regressor
k = 8 # Number of neighbors
knn_regressor = KNeighborsRegressor(n_neighbors=k)

# Train the regressor
model=knn_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn_regressor.predict(X_test)

# model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, y_pred)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred)) 
print('Root Mean Square Error (RMSE):',np.sqrt(mean_squared_error(y_test,y_pred)))


#Actual value and the predicted value
knn_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})

# saving the DataFrame as a CSV file 
gfg_csv_data = knn_model_diff.to_csv('outPut/knnAllScale.csv', index = True)


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



plt.scatter(y_test, y_pred, alpha=0.5)
plt.show()






#https://medium.com/towards-data-science/k-nearest-neighbors-94395f445221
knn_r_acc = []
for i in range(1,17,1):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)
    test_score = knn.score(X_test,y_test)
    train_score = knn.score(X_train,y_train)
    knn_r_acc.append((i, test_score ,train_score))
df_knn = pd.DataFrame(knn_r_acc, columns=['K','Test Score','Train Score'])

print(df_knn)








