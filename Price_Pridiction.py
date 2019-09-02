# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:34:04 2019

@author: aashique karn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing.csv') # It reads csv file
x = dataset.iloc[:,1:2].values # it will give all values of 1st column
y = dataset.iloc[:,4].values # it will give all values in 4th column
x1 = dataset.iloc[:,2:3].values
x2 = dataset.iloc[:,3:4].values
'''
# it will not work for this dataset because p value in second last column becomes 0 for all columnns
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((489,0)).astype(int),values=x,axis=1)
x_opt = x[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()  
'''
'''

#it is used to standarize out data.
# like we cannot compare between 50 out of 150 marks and 100 out of 200 marks
# to make it comparable we need to take percentage of both
#after that we know that one have got 50% and another have got 75%
from sklearn.preprocessing import StandardScaler
ST = StandardScaler()
x_train = ST.fit_transform(x_train)
x_test = ST.transform(x_test)
y_train = ST.fit_transform(y_train)
y_test = ST.transform(y_test)
'''

c = [x,x1,x2]

for z in c:

    from sklearn.model_selection import train_test_split
    #for testing only 10%  of data and spliting data into 2 parts
    x_train,x_test,y_train,y_test = train_test_split(z,y,test_size=0.10,random_state = 0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train,y_train) #it will fit model to our data
    y_pred = regressor.predict(x_test)   #predict that y_predict is same with x_test
    plt.scatter(x_train,y_train,color='red')
    plt.plot(x_train,regressor.predict(x_train),color='blue')
    plt.title("Linear Regression(Training set)")
    q = ["AVG NO. ROOMS/DWELLING(RM)","LOWER STATUS OF POPULATION(%)(LSTAT)","PUPIL TEACHER RATIO(PTRATIO)"]
    if z is x:
        plt.xlabel(q[0])
    elif z is x1:
        plt.xlabel(q[1])
    elif z is x2:
        plt.xlabel(q[2])
    else:
        print('Error')
    plt.ylabel('MEDV')    
    plt.show()
    
    
    plt.scatter(x_test,y_test,color='green')
    plt.plot(x_train,regressor.predict(x_train),color='red')
    plt.title("Linear Regression(Test set)")
    q = ["AVG NO. ROOMS/DWELLING(RM)","LOWER STATUS OF POPULATION(%)(LSTAT)","PUPIL TEACHER RATIO(PTRATIO)"]
    if z is x:
        plt.xlabel(q[0])
    elif z is x1:
        plt.xlabel(q[1])
    elif z is x2:
        plt.xlabel(q[2])
    else:
        print('Error')
    plt.ylabel('MEDV')    
    plt.show()

'''
# if any one of us have eager to work with 1st i.e 0th column of dataset
x0 = dataset.iloc[:,0:1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x0,y,test_size=0.10,random_state = 0)    
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)   
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Linear Regression(Training set)")
plt.xlabel('Rough Work')
plt.ylabel('MEDV')
plt.show()
plt.scatter(x_test,y_test,color='green')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Linear Regression(Test set)")
plt.xlabel('Rough Work')
plt.ylabel('MEDV')
plt.show()
'''
