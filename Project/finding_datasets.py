# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:01:04 2015

@author: karunsiddana
"""

import json
import urllib2
import urllib
import unirest

response = unirest.get("https://23andme-23andme.p.mashape.com/drug_responses/{profile_id}/",
  headers={
    "X-Mashape-Key": "SF6ODPg0UMmsh17ESJsL6WTGbK7Yp141HqPjsnqU4DRexBst5P",
    "Authorization": "<required>",
    "Accept": "text/plain"
  }
)

