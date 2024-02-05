# This file contains Chatbot class

# Import required libraries
import logging
from typing import List
import os
from dotenv import load_dotenv
from datetime import datetime
from functions import function_list
import openai
import pinecone
import json
import time
from llama_index.llms import OpenAI
from utils import get_embedding_MiniLM, search_news_api, write_dataframe_index, index, upsert_pinecone, write_dataframe
from actionweaver.llms.openai.chat import OpenAIChatCompletion
from actionweaver.llms.openai.tokens import TokenUsageTracker
import threading

logger = logging.getLogger(__name__)

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

# Inlitialize Pinecone DB
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
work_dir = "data"

def get_symbol_list():
    # Get available symbol list
    dirs = os.listdir(work_dir)
    symbols = []
    for d in dirs:
        if os.path.isdir(os.path.join(work_dir, d)):
            symbols.append(d)
    return symbols


class ChatBot():
    def __init__(self):
        # Initialize Chatbot
        self.logger = logger
        self.token_tracker = TokenUsageTracker(budget=None, logger=logger)
        self.llm = OpenAIChatCompletion(
            "gpt-3.5-turbo-16k", token_usage_tracker=self.token_tracker, logger=logger
        )
        self.index_name = "news-tracking"
        self.pinecone_index = pinecone.Index(self.index_name)
        self.top_k = 10
        self.work_dir = "data"
        self.model = OpenAI(model="gpt-3.5-turbo-16k")
        self.init_messages()
        self.message_templates = {
            "add" : r"New Symbol {} has been added. I will tracked real time.",
            "stop" : r"Symbol {} has been stoped.",
            "exist" : r"Symbol {} is already tracked by system.",
            "not_found" : r"Symbol {} is not tracked by system currently.",
            "remove" : r"Symbol {} is removed from system."
        }
        self.symbols = []
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
        self.t1 = threading.Thread(target=self.track_symbol)
        self.t1.start()


    def init_messages(self):
        # Initialize system message
        system_str = "You are a helpful assistant. Please do not try to answer the question directly."
        self.messages = [{"role": "system", "content": system_str}]


    def track_symbol(self):
        # Download news data for tracked symbols
        while True:
            time.sleep(10)
            for symbol in self.symbols:  
                symbol = symbol.lower()
                symbol_list = get_symbol_list()
                if symbol not in symbol_list:
                    print("Download the data for ", symbol)
                    for i in range(0, 25, 7):
                        df = search_news_api(symbol, i+7, i)
                        upsert_pinecone(index, df)
                        write_dataframe(df, "data", symbol)


    def run(self, query):
        # Generate response based on the input query
        db_response = self.query_db(query)      # Get the references from Pinecone DB
        context = (
            "Information from knowledge base:\n"
            "---\n"
            f"{db_response}\n"
            "---\n"
            f"User: {query}\n"
            "Answer question based on information from knowledge base. Your Response:"
        )
        self.messages.append({"role": "user", "content": context})
        if len(self.messages) > 3:
            ms = self.messages[-2:]
        else:
            ms = self.messages
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=ms,
            temperature=0,
            functions=function_list,
            function_call="auto",
        )       # Generate response
        if response.choices[0].finish_reason == "function_call":        # Check if function call
            param = json.loads(str(response.choices[0].message["function_call"]))
            if param["name"] == 'track_marketing_symbol':
                # print(param)
                symbol = json.loads(param["arguments"])['symbol']
                if symbol not in self.symbols:
                    self.symbols.append(symbol)
                    return self.message_templates["add"].format(symbol)
                else:
                    return self.message_templates["exist"].format(symbol)
            elif param["name"] == 'stop_tracking':
                symbol = json.loads(param["arguments"])['symbol']
                if symbol not in self.symbols:
                    return self.message_templates["not_found"].format(symbol)
                else:
                    self.symbols.remove(symbol)
                    return self.message_templates["remove"].format(symbol)
        else:
            return response.choices[0].message["content"]

    def query_db(self, query):
        # Query the Pinecone DB and get references
        query_vector = get_embedding_MiniLM(query)      # Calculate embedding vector for input query
        result = self.pinecone_index.query(vector=query_vector, top_k=self.top_k)       # Get the references from Pinecone DB
        matches = result.to_dict()["matches"]
        ids = []
        for match in matches:
            ids.append(match["id"])
        data = self.pinecone_index.fetch(ids).to_dict()["vectors"]
        descriptions = []
        for id in ids:
            descriptions.append(data[id]["metadata"]["description"])
        return descriptions

# bot = ChatBot()
# while True:
#     user_input = input("input query: ")
#     print('-----------')
#     result = bot.run(user_input)
#     print(result)