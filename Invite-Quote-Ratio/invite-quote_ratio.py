# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:39:03 2015

@author: karunsiddana
"""

import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('consumer_database.sqlite')

disk_engine = create_engine('sqlite:///consumer_database.sqlite')
df_requests = pd.read_sql_query('SELECT * FROM requests', disk_engine)
df_quotes = pd.read_sql_query('SELECT * FROM quotes', disk_engine)
df_invites = pd.read_sql_query('SELECT * FROM invites', disk_engine)
df_locations = pd.read_sql_query('SELECT * FROM locations', disk_engine)
df_categories = pd.read_sql_query('SELECT * FROM categories', disk_engine)

#Try seeing some samples and how the dataframes are related
df_quotes[df_quotes.invite_id == 4]
df_invites[df_invites.invite_id == 4]
df_requests[df_requests.request_id == 1]
df_locations[df_locations.location_id == 35]
df_categories[df_categories.category_id == 46]

#Now lets see find some trends in Photography
df_requests[df_requests.category_id == 1]
df_invites[df_invites.invite_id == 4]

#plot between request_id = 311
df_invites[df_invites.request_id == 311].groupby('request_id').plot(kind='bar')

#Merge dataframes using inner join on the Location dataframe and the request_dataframe
frame = pd.DataFrame.merge(df_requests, df_locations, left_on='location_id', right_on='location_id')

#MErge Categories and Location Dataframe into one dataframe using category_id as the key
frame1 = pd.DataFrame.merge(frame, df_categories, left_on='category_id', right_on='category_id')

#Merge Invite Dataframe with the Request Dataframe using the request_id as the key do an inner join
frame2 = pd.DataFrame.merge(frame1, df_invites, left_on='request_id', right_on='request_id')

#Merge Quotes dataframe to the Final Dataframe including Requests, Invites and time 
#using the invite_id as the Key for the inner join.
final_df = pd.DataFrame.merge(frame2, df_quotes, left_on='invite_id', right_on='invite_id')
#Index([u'request_id', u'user_id_x', u'category_id', u'location_id', u'creation_time', 
#u'name_x', u'name_y', u'invite_id', u'user_id_y', u'sent_time_x', 
#u'quote_id', u'sent_time_y'], dtype='object')

#Format the Columns in to Dates and Time Columns
import matplotlib.dates as DT

final_df['creation_time_test'] = pd.to_datetime(final_df.creation_time, "%Y-%M-%D")

final_df['c_date'] = final_df['creation_time'].map(lambda x: x[:10])
final_df['c_time'] = final_df['creation_time'].map(lambda x: x[11:-7])
final_df['c_date2num'] = final_df['c_date'].map(lambda x: DT.datestr2num(x))

final_df['i_date'] = final_df['sent_time_x'].map(lambda x: x[:10])
final_df['i_time'] = final_df['sent_time_x'].map(lambda x: x[11:-7])
final_df['i_date2num'] = final_df['i_date'].map(lambda x: DT.datestr2num(x))

final_df['q_date'] = final_df['sent_time_y'].map(lambda x: x[:10])
final_df['q_time'] = final_df['sent_time_y'].map(lambda x: x[11:-7])
final_df['q_date2num'] = final_df['q_date'].map(lambda x: DT.datestr2num(x))

final_df.drop(['creation_time', 'sent_time_x', 'sent_time_y'], axis=1, inplace=True)

#This is the relationship that we are looking for
final_df[['i_time', 'q_time']]
final_df[['i_date','q_date','i_time', 'q_time']]

#Plot some Line Graphs
plt.plot(final_df.q_date2num, final_df.category_id )
plt.plot_date(final_df.i_date2num, final_df.category_id)
plt.plot_date(final_df.i_date2num.head(20), final_df.category_id.head(20))

#Trying to plot the Quote Time and the Invite DateTime on the same Axis and see what the difference is
#Can I plot a histogram instead? What will give me the best insight to my data?
#Bar Graph?






'''
c = conn.cursor()
c.execute('SELECT SQLITE_VERSION()')
version = c.fetchone()

print "SQLite Version is %s" %version

c.execute('SELECT * from categories')
c.fetchone()

c.execute('SELECT * from invites')
c.fetchone()

c.execute('SELECT * from locations')
c.fetchone()

c.execute('SELECT * from quotes')
c.fetchone()

c.execute('SELECT * from requests')
c.fetchone()

c.execute('SELECT * from users')
c.fetchone()

users = pd.read_sql_table('requests', conn)
'''
