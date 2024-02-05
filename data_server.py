# This file contains the flask server that serve news data to frontend

# Import required libraries
from fastapi import FastAPI
import os
import json
from datetime import datetime
from server import ChatBot
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# FastAPI server
app = FastAPI()

# Cors Setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Chatbot
bot = ChatBot()

work_dir = "data"


def format_date(date_string):
    # Transfer the date to DD%MM format
    date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    day = date.day
    month = date.strftime("%b")
    return f"{day} {month}"

def global_sentiment(sentiment):
    # Transfer the sentiment to number
    if "neutral" in sentiment:
        return 0
    elif "positive" in sentiment:
        return 1
    elif "negative" in sentiment:
        return -1
    else:
        return 0

@app.get('/')
def home():
    # Test Url
    return "Hello"

@app.get('/symbol_list')
def get_symbol_list():
    # Get available symbol list
    dirs = os.listdir(work_dir)
    symbols = []
    for d in dirs:
        if os.path.isdir(os.path.join(work_dir, d)):
            symbols.append(d)
    return symbols

@app.get('/get_all')
def get_all(start, end):
    # Get all available news data
    symbols = get_symbol_list()
    res = []
    for symbol in symbols:
        res.append(get_news(symbol, start, end))
    return res

@app.get('/news/{symbol}')
def get_news(symbol, start, end):
    # Get news data for specific symbol from start date to end date
    sym_list = get_symbol_list()
    start = datetime.strptime(start, '%Y-%m-%dT%H:%M:%SZ')
    end = datetime.strptime(end, '%Y-%m-%dT%H:%M:%SZ')
    result = []
    final_data = {}
    final_data["name"]= symbol
    neg = 0
    pos = 0
    net = 0
    if symbol not in sym_list:
        print("symbol not found")
        final_data["data"] = []
        final_data["count"] = {
            "name": symbol,
            'data':[neg, pos, net]
        }
        return final_data
    else:
        articles = os.listdir(os.path.join(work_dir, symbol))
        symbol_dir = os.path.join(work_dir, symbol)
        for article in articles:
            date = article.split(" ")[0]
            date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
            if date > start and date < end:
                with open(os.path.join(symbol_dir, article)) as f:
                    data = json.load(f)
                    try:
                        result.append([data["publishedAt"], global_sentiment(data["sentiment"])])

                        if global_sentiment(data["sentiment"]) ==0:
                            net +=1
                        elif global_sentiment(data["sentiment"])==-1:
                            neg+=1
                        elif global_sentiment(data["sentiment"]) ==1:
                            pos+=1
                    except:
                        print(data.keys())
        final_data["data"] = result
        final_data["count"] = {
            "name":symbol,
            'data':[neg, pos, net, neg+pos+net]
        }
        return final_data



@app.get('/ask')
def ask(query):
    # Generate the answer from the bot
    result = bot.run(query)
    return result