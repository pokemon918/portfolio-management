# Import required Libraries
import openai
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index.storage import StorageContext
from llama_index.llms import OpenAI
from utils import get_embedding_MiniLM


# Import Environment Variables
openai.api_key = "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"
PINECONE_API_KEY = "55190211-17de-4966-a316-ca4604288a27"
PINECONE_ENV = "gcp-starter"
NEWS_API_KEY = "1e2adc7d46ca4d8d92e0fbf88fee01b3"

# Initiate Pinecone Database
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Chatbot Agent
class ChatBot:

    def __init__(self, index_name = "news-tracking", model="gpt-3.5-turbo-16k"):
        # This function initiate Chatbot Agent
        self.index_name = index_name
        self.pinecone_index=pinecone.Index(self.index_name)
        self.top_k = 10
        self.model = OpenAI(model=model)
        self.init_messages()

    def init_messages(self):
        # This function initialize chat messages
        system_str = "You are a helpful assistant. Please do not try to answer the question directly."
        self.messages = [{"role": "system", "content": system_str}]    

    def run(self, query):
        # This function generate response based on query
        db_response = self.query_db(query)          # Get references from the Pinecone DB
        context = (
            "Information from knowledge base:\n"
            "---\n"
            f"{db_response}\n"
            "---\n"
            f"User: {query}\n"
            "Only answer question based on information from knowledge base"
            "If you don't have information in the knowledge base, please response as \"I can not find the related information from knowledge base.\". Your Response:"
        )
        self.messages.append({"role":"user", "content":context})        # Initialize input messages
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=self.messages,
            temperature=0,
        )       # Generate response based on message
        return response.choices[0].message["content"]

    def query_db(self, query):
        # This function query the Pinecone DB and get the references
        query_vector = get_embedding_MiniLM(query)      # Get the embedding vector of input query
        result = self.pinecone_index.query(vector=query_vector, top_k=self.top_k)       # Query the Pinecone DB and get the reference
        matches = result.to_dict()['matches']
        ids = []
        for match in matches:
            ids.append(match['id'])
        data = self.pinecone_index.fetch(ids).to_dict()["vectors"]
        descriptions = []
        for id in ids:
            print(data[id]['metadata'].keys())
            descriptions.append(data[id]['metadata']['description'])
        return descriptions