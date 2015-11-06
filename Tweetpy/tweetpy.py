# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:01:04 2015

@author: karunsiddana
"""

import tweepy

# Consumer keys and access tokens, used for OAuth
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    if  tweet.text.find("#India"):
        #print tweet.user
        print tweet.user.name
        print tweet.text
    
user = api.me()
print user

def on_status(self, status):
        # Prints the text of the tweet
        print('Tweet text: ' + status.text)
 
        # There are many options in the status object,
        # hashtags can be very easily accessed.
        for hashtag in status.entries['hashtags']:
            print(hashtag['text'])
 
        return True

# Sample method, used to update a status
api.update_status('Hello Python Central!')

trends_available = api.trends_available()
for each_trend in trends_available:
    print each_trend['country']
    #print each_trend

for tweet in api.search("@myfitnesspal"):
    print tweet.text
    print tweet.user.name
    print tweet.user.location
    print tweet.user.created_at
    print "\n"
    
for tweet in api.search("#TripAdvisor"):
    print tweet.text
    print tweet.user.name
    print tweet.user.location
    print tweet.user.created_at
    print "\n"
    
