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
    #EmailFetcher class
class EmailFetcher:
    def __init__(self):
        load_dotenv()
        self.service = self.gmail_authenticate()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.email_query = os.getenv("EMAIL_QUERY")
        self.prompt = (
            "You are given the plain text content of more than one newsletter email. Your task is to extract and return only "
            "the title and the first paragraph of the main news content, ensuring that all ads and sponsored content are removed. "
            "Since the email contains more than one newsletter, you must extract all of them. Follow these steps:\n"
            "    1. Identify the title of the main news content.\n"
            "    2. Extract the first paragraph of the main news content.\n"
            "    3. Remove any ads, sponsored content, or unrelated promotional material.\n"
            "    4. Return a list of all news items, with the output format for each news being:\n"
            '        "Content: {content}\n'
            '        "----------------------"\n'
            "    Return the output as plain text."
        )
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def gmail_authenticate(self):
        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json',
                    ['https://www.googleapis.com/auth/gmail.readonly']
                )
                creds = flow.run_local_server(port=0)
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)
        return build('gmail', 'v1', credentials=creds)
        