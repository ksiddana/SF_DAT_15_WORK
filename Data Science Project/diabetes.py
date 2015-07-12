# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:01:04 2015

@author: karunsiddana
"""

# read data into a DataFrame
obesity_data = pd.read_csv('diabetes_obesity.csv', index_col=0)
obesity_data.columns
# scatter matrix in Seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(obesity_data)

#Out[42]: Index([u'County Code', u'Region Name', u'Indicator Number',
#u'Indicator', u'Total Event Counts', u'Denominator', u'Denominator Note', 
#u'Measure Unit', u'Percentage/Rate', u'95% CI', u'Data Comments', u'Data Years', 
#u'Data Sources', u'Quartile', u'Mapping Distribution', u'Location'], dtype='object')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()

# read diabetes_vanderbilt.csv data into a DataFrame
import pandas as pd
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


#Visual Plot of Barium is only seen in Headlamps
#diabetes_data.boxplot(column='stab.glu', by='glyhb', rot=90)
#plotting the Sodium against Refractive Index
#diabetes_data.plot(x="glyhb", y="stab.glu")

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


# cross-validation train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=.33, random_state=1)

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

# transform predictions to 1 or 0
assorted_pred_class = np.where(assorted_pred >= 7, 1, 0)
assorted_pred_class

# add predicted class to DataFrame
y['assorted_pred_class'] = assorted_pred_class

# scatter plot that includes the regression line
plt.scatter(X_train, y_train)
plt.plot(X_train, assorted_pred_class, color='red')

# sort DataFrame by al
y.sort('glyhb', inplace=True)

# make predictions on testing set and calculate accuracy
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


##-----------------------------------
##Linear Regeression Model
##-----------------------------------

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x, y)

# look at the coefficients to get the equation for the line, but then how do you plot the line?
print linreg.intercept_
print linreg.coef_

linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print metrics.accuracy_score(y_test, y_pred)

# scatter matrix in Pandas
pd.scatter_matrix(x, figsize=(12, 10))
