import json
import pytz
from datetime import datetime, timedelta
from ntscraper import Nitter
import pandas as pd
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from bs4 import BeautifulSoup
from base64 import urlsafe_b64decode
import google.generativeai as genai
import unicodedata
import re
import string
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from  vector_db.qdrant_db import Qdrant




#TwitterFetcher class
class TwitterFetcher:
    def __init__(self):
        load_dotenv()
        self.twitter_users = json.loads(os.getenv("TWITTER_USERS").replace("'", '"'))
        self.scraper = Nitter(log_level=0, skip_instance_check=False)
        self.scraper.instance = "https://nitter.woodland.cafe/"

    def fetch_24_hour_tweets(self):
        timezone = pytz.timezone("UTC")
        current_time = datetime.now(timezone)
        last_24_hours = current_time - timedelta(hours=24)

        data = {
            'text': [],
            'date': [],
            'retweets': [],
            'quoted-tweets': []
        }

        for user in self.twitter_users:
            print(f"Fetching tweets for user: {user}")
            try:
                tweets = self.scraper.get_tweets(user, mode="user")
            except Exception as e:
                print(f"Error fetching tweets for user {user}: {e}")
                continue

            if tweets['tweets']:
                for tweet in tweets['tweets']:
                    tweet_time = datetime.strptime(tweet['date'], '%b %d, %Y · %I:%M %p %Z').replace(tzinfo=pytz.utc).astimezone(timezone)
                    if tweet_time >= last_24_hours:
                        data['text'].append(tweet['text'])
                        data['date'].append(tweet_time.strftime('%Y-%m-%d %H:%M:%S %Z%z'))

                        if tweet['is-retweet']:
                            data['retweets'].append(tweet['link'])
                        else:
                            data['retweets'].append("No retweet")

                        if 'quoted-post' in tweet:
                            quoted_info = tweet['quoted-post']
                            quoted_text = quoted_info.get('text', 'No text available')
                            data['quoted-tweets'].append(quoted_text)
                        else:
                            data['quoted-tweets'].append("No quoted tweet")

            else:
                print(f"No tweets were found for user: {user}.")

        df = pd.DataFrame(data)
        print(df)
        return df
        