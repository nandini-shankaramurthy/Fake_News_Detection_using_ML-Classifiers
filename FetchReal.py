

import os
from sys import path
import csv
from textblob import TextBlob

import requests
from bs4 import BeautifulSoup
from newscrape_common import (is_string, ist_to_utc, remove_duplicate_entries,str_is_set)
from sources import KNOWN_NEWS_SOURCES
import json
import re

import tweepy    #this will give an error if tweepy is not installed properly
from tweepy import OAuthHandler
 
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


result=[]

def sentiment(tweet):
        analysis = TextBlob(tweet)
       
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'positive'
        else:
            return 'negative'   



def get_all_content(objects):
    """
    Call this function with a list of objects. Make sure there are no duplicate
    copies of an object else downloading might take long time.
    """
    def get_content(url):
        response = requests.get(url)
        if response.status_code == 200:
            html_content = BeautifulSoup(response.text, "html.parser")
            # a bit erraneous
            # contents sometimes include unwanted text when they too are defined in p tag
            contents = html_content.find('div', {'id': 'storyBody'}
                        ).find_all(lambda tag: tag.name == 'p' and not tag.img, recursive=False)
            text = ''
            for cont in contents:
                text += cont.get_text() + '\n'
            return text
        return "NA"

    for obj in objects:
        obj["content"] = get_content(obj["link"])
        print(obj["link"])
        print(obj["title"])
        print("---------------------------------------------") 
        print(obj["content"])
        print("---------------------------------------------")
        dd=sentiment(obj["content"])

        result.append([obj["content"],"REAL",dd])

def get_headline_details(obj):
    try:
        from datetime import datetime
        timestamp_tag = obj.find(
            "span", {"class": "SunChDt2"}
        )
        if timestamp_tag is None:
            timestamp = datetime.now()
        else:
            content = timestamp_tag.contents[0].strip()
            #print(content)
            timestamp = datetime.strptime(content,"%d %b %Y %I:%M %p")
        return {
            "content": "NA",
            "link": "https://www.deccanchronicle.com" + obj["href"],
            "scraped_at": datetime.utcnow().isoformat(),
            "published_at": ist_to_utc(timestamp).isoformat(),
            "title": obj.find(['h3', 'h2']).contents[0].strip()
        }
    except KeyError:
        import pdb
        pdb.set_trace()


def get_chronological_headlines(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        a_tags = list(
                map( lambda x: x.find("div", {"class": "col-sm-8"}).find("a"),
                    soup.find_all("div", {"class": "col-sm-12 SunChNewListing"})
                    )
                )
        headlines = list(map(get_headline_details, a_tags))
        get_all_content(headlines)  # Fetch contents separately
        return headlines
    return None


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

def clean_tweet(tweet):
	return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def process():
	SRC = KNOWN_NEWS_SOURCES["Deccan Chronicle"]
	for j in range(0,5):
		print(json.dumps(get_chronological_headlines(SRC["pages"].format(j)),sort_keys=True,indent=4))
	print(result)
	with open('data.csv', 'w', newline='',encoding="utf-8") as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(["text","label","Sentiment"])
		for j in result:
			wr.writerow(j)
	twitter_stream = Stream(auth, MyListener())
	#change the keyword here
	twitter_stream.filter(track=['#Apple'])
	
	with open('twitter.json') as in_file, open('data.csv', 'a',newline='') as out_file:
		writer = csv.writer(out_file, delimiter=',')
		for line in in_file:
			if line.strip() == "" :
				continue
			else:
				tweet = json.loads(line)
				tweet['text']=clean_tweet(tweet['text'])
				dd=sentiment(tweet['text'])

				print(tweet['text'])
				row = (tweet['text'],"Fake",dd)
				writer.writerow(row)
	out_file.close()

		

