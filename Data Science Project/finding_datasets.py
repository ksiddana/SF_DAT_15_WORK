# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:01:04 2015

@author: karunsiddana
"""

import json
import urllib2
import urllib
import unirest
import pandas as pd

response = unirest.get("https://23andme-23andme.p.mashape.com/profile_picture/a42e94634e3f7683/",
  headers={
    "X-Mashape-Key": "ElJZ9z80uzmshjGaMAkzm77qIYtKp14zeyOjsndLYpXr97rURe",
    "Authorization": "[OAuth-HTTP-MAC]",
    "Accept": "text/json"
  }
)

file = open("maryland.json",'r')
data = json.load(file)

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
