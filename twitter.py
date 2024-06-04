
import tweepy    #this will give an error if tweepy is not installed properly
from tweepy import OAuthHandler
import sys
import json
import csv
import re
 
#provide your access details below 
consumer_key = "HyvTGPZOElNuzFcvkWG2F2FQv"
consumer_secret = "mIsUnLgWVCVLUrbIbvP5QnzHjCsvMhKVCnmM7A2sbWeTNnMhP8"
access_token = "964365971887501312-TlreD3J7RkMshqx8qYbNIIZ6kBjQUBO"
access_token_secret = "NPfG72WkzFyEyd5DwcX9fDeH6rrMbAeqYkLBE6wID7R7m"
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
api = tweepy.API(auth)    
    
from tweepy import Stream
from tweepy.streaming import StreamListener
 
i=0

class MyListener(StreamListener):

    def on_data(self, data):
        global i
        try:
            
            with open('twitter.json', 'a') as f:  #change location here
            	i=i+1
            	print(i)
            	f.write(data)
            	if i>2:
            		sys.exit(1)
            		return False
            	else:
            		return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            return False
 
    def on_error(self, status):
        print(status)
        return False

twitter_stream = Stream(auth, MyListener())

#change the keyword here
twitter_stream.filter(track=['#Apple'])

def clean_tweet(tweet):
	return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


with open('twitter.json') as in_file, open('data.csv', 'a',newline='') as out_file:
    writer = csv.writer(out_file, delimiter=',')
    for line in in_file:
        if line.strip() == "" :
               	continue
        else:
        	tweet = json.loads(line)
        	tweet['text']=clean_tweet(tweet['text'])
        	print(tweet['text'])
        	
        	row = (tweet['text'],"Fake")    	
        	values = [(value.encode('utf8') if hasattr(value, 'encode') else value) for value in row]
        	writer.writerow(row)
out_file.close()
