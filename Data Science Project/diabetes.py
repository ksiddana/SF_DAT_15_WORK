# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:01:04 2015

@author: karunsiddana
"""

# read diabetes_vanderbilt.csv data into a DataFrame
import pandas as pd
import numpy as np
diabetes_data = pd.read_csv('diabetes_vanderbilt.csv', index_col=0)
diabetes_data.columns
#Index([u'chol', u'stab.glu', u'hdl', u'ratio', u'glyhb', u'location', 
#u'age', u'gender', u'height', u'weight', u'frame', u'bp.1s', u'bp.1d', 
#u'bp.2s', u'bp.2d', u'waist', u'hip', u'time.ppn'], dtype='object')

diabetes_data.gender.value_counts()
#female    234
#male      169

import seaborn as sns
import matplotlib.pyplot as plt
# scatter matrix in Seaborn
diabetes_data.plot(x="hdl",y="glyhb")
plt.scatter(diabetes_data.hdl, diabetes_data.chol, c='red')
sns.lmplot(x='hdl', y='glyhb', data=diabetes_data, ci=None)
sns.lmplot(x='stab.glu', y='glyhb', data=diabetes_data, ci=None)

#Kernel desnsity plots from 
#http://web.stanford.edu/~mwaskom/software/seaborn/tutorial/distributions.html
sns.jointplot(x="glyhb", y="stab.glu", data=diabetes_data, kind="kde")


features = ['chol', 'stab.glu', 'hdl']
x = diabetes_data[features]
x
y = diabetes_data['glyhb']
y

#find the missing data from the datasets and try filling them out
x.chol.isnull()
x.chol.fillna(x.chol.mean(), inplace=True)
x.hdl.isnull()
x.hdl.fillna(40, inplace=True)
#the output is NaN for 13 datapoints, therefore we need to fill in the data
y.isnull().value_counts()
#Out[18]: 
#False    390
#True      13
y.fillna(5.5, inplace=True)
y_category = np.where(y > 7, 1, 0)

# cross-validation train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y_category,test_size=.33, random_state=1)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

##-----------------------------------
##Logistic Regeression Model
##-----------------------------------

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
zip(features, logreg.coef_[0])

#look at the coefficients to get the equation for the line, 
#but then how do you plot the line?
print logreg.intercept_
print logreg.coef_

assorted_pred = logreg.predict(X_train)
assorted_pred[:20]

# add predicted class to DataFrame
#y['assorted_pred_class'] = assorted_pred_class

# scatter plot that includes the regression line
plt.scatter(X_train, y_train)

# sort DataFrame by glyhb
y.sort('glyhb', inplace=True)

# make predictions on testing set and calculate accuracy
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


##-----------------------------------
##Linear Regeression Model
##-----------------------------------

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.33, random_state=1)


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# look at the coefficients to get the equation for the line, but then how do you plot the line?
print linreg.intercept_
print linreg.coef_

#predict the values of an Out-of Sample test Data based on X_test
y_pred = linreg.predict(X_test)

# MAE is the same as before
# Calculate the error between the predicted values of y_pred and y_test
print metrics.mean_absolute_error(y_test, y_pred)

# RMSE is larger than before
# Calculate the error between the predicted values of y_pred and y_test
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(linreg, x, y, cv=5)
print scores

# scatter matrix in Pandas
pd.scatter_matrix(x, figsize=(12, 10))
