'''
Created on 31 Jul 2014

@author: edwin
'''
import urllib, urllib2

url = "http://localhost:8000/crowdReports"

values = {'c_val' : '0',
          'type': '0',
          'lat' : '18.55',
          'lng' : '-72.3' }

#These lines send the data as a form not a query string
data = urllib.urlencode(values)
req = urllib2.Request(url, data)
response = urllib2.urlopen(req)
the_page = response.read()
print str(the_page)