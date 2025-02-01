import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
DATASET_PATH = os.getenv('DATASET_PATH')
SHEET_NAME = os.getenv('SHEET_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')