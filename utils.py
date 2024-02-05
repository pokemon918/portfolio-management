# This file contains utilities for Chatbot

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS
import yfinance as yf
from dotenv import load_dotenv
import os
import pinecone
import uuid
import numpy as np
import json

from datetime import datetime, timedelta
import openai
from openai.embeddings_utils import get_embedding
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from newsapi import NewsApiClient

from sentence_transformers import SentenceTransformer
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Environment variables
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# openai.api_key = "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"
# PINECONE_API_KEY = "55190211-17de-4966-a316-ca4604288a27"
# PINECONE_ENV = "gcp-starter"
# NEWS_API_KEY = "1e2adc7d46ca4d8d92e0fbf88fee01b3"


embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)    # Load Embedding model
embedding_model = "text-embedding-ada-002"    # Openai Embedding model
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000       # Openai Embedding encode length

nltk.download("punkt")
user_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0"
)

config = Config()       # Configure for Newspaper library
config.browser_user_agent = user_agent
config.request_timeout = 10

# Load Models
base_model = "THUDM/chatglm2-6b"        # FinGPT base model
peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"       # FinGPT Peft Model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)       # FinGPT tokenizer
model = AutoModel.from_pretrained(base_model, trust_remote_code=True, device_map="auto")        # Load FinGPT base model
model = PeftModel.from_pretrained(model, peft_model)        # Load FinGPT
model = model.eval()        # Initialize FinGPT as evaluation model

newsapi = NewsApiClient(api_key=NEWS_API_KEY)       # Initialize News api

nltk.download("vader_lexicon")  # required for Sentiment Analysis

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)       # Initialize Pinecone DB

index_name = "news-tracking"        # Name of Used Pinecone index

# # only create index if it doesn't exist
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         name=index_name,
#         dimension=1536,
#         metric='cosine'
#     )
#     # wait a moment for the index to be fully initialized
#     time.sleep(1)

# # now connect to the index
index = pinecone.GRPCIndex(index_name)      # Load the Pinecone Index

def get_embedding_MiniLM(query):
    # Calculate Embedding of Input Query
    xq = embed_model.encode(query).tolist()
    return xq


def get_sentiment(sentence):
    # Get the sentiment of input sentences
    prompts = []
    prompt = """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: {}
Answer: """.replace(
                "{}", sentence
            )
    prompts.append(prompt)
    tokens = tokenizer(prompts, return_tensors="pt", padding=True, max_length=512)
    for key in tokens.keys():
        tokens[key] = tokens[key].cuda()
    res = model.generate(**tokens)
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    return out_text[0]


def sentiment_batch(batch):
    # Get the sentiment for batch of sentences
    tokens = tokenizer(batch, return_tensors="pt", padding=True, max_length=512)
    for key in tokens.keys():
        tokens[key] = tokens[key].cuda()
    res = model.generate(**tokens)
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    return out_text


def analyze_sentiment(news_list):
    # Calculate sentiment for input news data
    prompts = []
    for news in news_list:
        prompt = """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: {}
Answer: """.replace(
            "{}", news
        )
        prompts.append(prompt)
    final_result = []
    length = len(prompts)
    num = len(prompts) // 32
    for i in range(num):
        prompt_batch = prompts[i*32:(i+1)*32]
        out_text = sentiment_batch(prompt_batch)
        final_result+=out_text
    final_result += sentiment_batch(prompts[num * 32: length])

    return final_result


def search_google_news(query, start=1, end=0):
    # Gather news data using Google news API
    start_date = dt.date.today() - dt.timedelta(days=start)
    start_date = start_date.strftime("%Y-%m-%d")

    end_date = dt.date.today() - dt.timedelta(days=end)
    end_date = end_date.strftime("%Y-%m-%d")

    print(start_date, end_date)
    googlenews = GoogleNews()
    googlenews.set_time_range(start_date, end_date)
    googlenews.search(query)
    result = googlenews.result()
    df = pd.DataFrame(result)

    return df


def search_yahoo_news(ticker):
    # Gather news data using yahoo finance API
    msft = yf.Ticker(ticker)
    result = []
    for i in msft.news:
        dic = {}
        article = Article(i["link"], config=config)  # providing the link
        try:
            article.download()  # downloading the article
            article.parse()  # parsing the article
            article.nlp()  # performing natural language processing (nlp)
        except:
            pass
        dic["Title"] = article.title
        dic["Article"] = article.text
        dic["Summary"] = article.summary
        dic["Key_words"] = article.keywords
        dic["Date"] = i["providerPublishTime"]
        dic["Publisher"] = i["publisher"]
        dic["Embedding"] = get_embedding(article.summary, engine=embedding_model)
        result.append(dic)

    result = pd.DataFrame(result)
    return result


def search_news_api(query, start=2, end=1, only_last = False):
    # Gather news Data using newsAPI
    start_date = dt.date.today() - dt.timedelta(days=start)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = dt.date.today() - dt.timedelta(days=end)
    end_date = end_date.strftime("%Y-%m-%d")
    articles = newsapi.get_everything(
        q=query, language="en", from_param=start_date, to=end_date
    )
    result = []
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=45)
    for i in articles["articles"]:
        try:
            i["embedding"] = get_embedding_MiniLM(i["description"])
            result.append(i)
        except:
            pass
    result = pd.DataFrame(result)
    description = result["description"].to_list()
    sentiment = analyze_sentiment(description)
    result["sentiment"] = sentiment
    return result


def filter_last_hour(news):
    # Filter News data for first 1 hour
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    result = []
    for article in news['articles']:
        # Adjust this line depending on the date-time format:
        published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        if published_at >= one_hour_ago:
            result.append(article)
    return result


def write_dataframe(data, work_dir, symbol):
    # Save news data to local directory
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    if not os.path.exists(os.path.join(work_dir, symbol)):
        os.mkdir(os.path.join(work_dir, symbol))
    source_dir = os.path.join(work_dir, symbol)
    data = data.to_dict("records")
    for record in data:
        file_name = str(record['publishedAt']) + " " + str(record['title'])
        file_name = file_name.replace("/", "_").replace("\\", "_")
        file_path = os.path.join(source_dir, file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump(record, file)


def write_dataframe_index(data, work_dir, symbol, index=index):
    # Save news data to Pinecone DB
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    if not os.path.exists(os.path.join(work_dir, symbol)):
        os.mkdir(os.path.join(work_dir, symbol))
    source_dir = os.path.join(work_dir, symbol)
    item_ids = []
    # embeddings = df['embedding'].tolist()
    embeddings = []
    metadata = []
    data = data.to_dict("records")
    for record in data:
        file_name = record['publishedAt'] + " " + record['title']
        file_name = file_name.replace("/", "_").replace("\\", "_")
        file_path = os.path.join(source_dir, file_name)
        if not os.path.exists(file_path):
            if (
            record["author"] is not None
            and record["title"] is not None
            and record["description"] is not None
            and record["url"] is not None
            and record["publishedAt"] is not None
            and record["content"] is not None
        ):
                item_ids.append(str(uuid.uuid4()))
                embeddings.append(record["embedding"])
                del record["embedding"]
                del record["urlToImage"]
                del record["source"]
                metadata.append(record)
            with open(file_path, 'w') as file:
                json.dump(record, file)
    if len(item_ids) > 0:
        records = zip(item_ids, embeddings, metadata)
        upsert_results = index.upsert(vectors=records)


def upsert_pinecone(index, df):
    # Insert Dataframe to Pinecone index
    item_ids = []
    embeddings = []
    metadata = []
    df = df.to_dict("records")
    for i, metadata_i in enumerate(df):
        if (
            metadata_i["author"] is not None
            and metadata_i["title"] is not None
            and metadata_i["description"] is not None
            and metadata_i["url"] is not None
            and metadata_i["publishedAt"] is not None
            and metadata_i["content"] is not None
        ):
            item_ids.append(str(uuid.uuid4()))
            embeddings.append(metadata_i["embedding"])

            del metadata_i["embedding"]
            del metadata_i["urlToImage"]
            del metadata_i["source"]

            metadata.append(metadata_i)

    records = zip(item_ids, embeddings, metadata)
    upsert_results = index.upsert(vectors=records)
    return upsert_results


# for i in range(0, 365, 7):
#     df = search_news_api("microsoft", i+7, i)
#     upsert_pinecone(index, df)
#     write_dataframe(df, "data", "microsoft")
#     df = search_news_api("google", i+7, i)
#     upsert_pinecone(index, df)
#     write_dataframe(df, "data", "google")
#     df = search_news_api("apple", i+7, i)
#     upsert_pinecone(index, df)
#     write_dataframe(df, "data", "apple")