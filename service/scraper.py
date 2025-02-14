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
    
    def search_messages(self, query):
        result = self.service.users().messages().list(userId='me', q=query).execute()
        messages = []
        if 'messages' in result:
            messages.extend(result['messages'])
        while 'nextPageToken' in result:
            page_token = result['nextPageToken']
            result = self.service.users().messages().list(userId='me', q=query, pageToken=page_token).execute()
            if 'messages' in result:
                messages.extend(result['messages'])
        return messages
    
    def get_content(self, text):
        genai.configure(api_key=self.api_key)
        response = self.model.generate_content([self.prompt, text])
        content = response.text
        return content

    def parse_parts(self, parts):
        if not parts:
            return ''
        
        text = ''
        for part in parts:
            mimeType = part.get("mimeType")
            body = part.get("body")
            data = body.get("data")
            if part.get("parts"):
                text += self.parse_parts(part.get("parts"))
            if mimeType == "text/plain":
                if data:
                    text += urlsafe_b64decode(data).decode()
            elif mimeType == "text/html":
                if data:
                    text += urlsafe_b64decode(data).decode()
                    text = BeautifulSoup(text, "html.parser").get_text()
        return text
    
    def read_message(self, message):
        msg = self.service.users().messages().get(userId='me', id=message['id'], format='full').execute()
        payload = msg['payload']
        headers = payload.get("headers")
        parts = payload.get("parts")
        email_data = {
            'from': '',
            'title': '',
            'content': ''
        }
        if headers:
            for header in headers:
                name = header.get("name")
                value = header.get("value")
                if name.lower() == 'from':
                    email_data['from'] = value
                elif name.lower() == 'subject':
                    email_data['title'] = value
        text = self.parse_parts(parts)
        email_data['content'] = self.get_content(text)
        return email_data

    def fetch_emails(self):
        results = self.search_messages(self.email_query)
        print(f"Found {len(results)} results.")
        emails = []
        for index, msg in enumerate(results):
            email_data = self.read_message(msg)
            email_data['index'] = index
            emails.append(email_data)
        df = pd.DataFrame(emails, columns=['index', 'from', 'title', 'content'])
        print(df)
        return df


#TextProcessor class
class TextProcessor:
    def __init__(self, email_df, tweet_df, model_name='all-MiniLM-L6-v2'):
        self.email_df = email_df
        self.tweet_df = tweet_df
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.df = None

    def process_text(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text

    def prepare_data(self):
        self.email_df['total_content'] = self.email_df['from'] + self.email_df['title'] + self.email_df['content']
        self.tweet_df['total_content'] = self.tweet_df['text'] + self.tweet_df['retweets'] + self.tweet_df['quoted-tweets']

        self.df = pd.concat([self.email_df[['total_content']], self.tweet_df[['total_content']]], axis=0).reset_index(drop=True)

    def add_processed_content(self):
        self.df['processed_content'] = self.df['total_content'].apply(self.process_text)

    def generate_embeddings(self):
        self.df['processed_content'] = self.df['processed_content'].fillna('')
        processed_content = self.df['processed_content'].tolist()
        all_embeddings = self.model.encode(processed_content)
        self.df['embedding'] = list(all_embeddings)

    def get_dataframe(self):
        return self.df
    
    
def scraper():
    
    twitter_fetcher = TwitterFetcher()
    tweet_df = twitter_fetcher.fetch_24_hour_tweets()
    
    email_fetcher = EmailFetcher()
    email_df = email_fetcher.fetch_emails()
    
    
    processor = TextProcessor(email_df=email_df, tweet_df=tweet_df)
    processor.prepare_data()
    processor.add_processed_content()
    processor.generate_embeddings()
    processed_df = processor.get_dataframe()
    processed_df['id'] = processed_df.index


 
    payloads = processed_df['processed_content']
    
    qdrant = Qdrant()

    collection_name = 'trend'
    vector_size = 384
    data= processed_df
    qdrant.items_embedding_batch_insert_data(vector_size,collection_name, data, batch_size=500)
    
  
 
 
    

        