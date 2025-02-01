import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from langchain_core.documents import Document
import os
from pinecone import Pinecone, ServerlessSpec
# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Now do stuff
if os.environ.get("PINECONE_INDEX_NAME") not in pc.list_indexes().names():
        pc.create_index(
            name=os.environ.get("PINECONE_INDEX_NAME"), 
            dimension=768, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

# Initialize embedding model
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Pinecone vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

def store_embeddings(labeled_data):
    docs = []

    for item in labeled_data:
        categories = item.get("categories", "")
        print(categories)
        scores = item.get("scores", "")
        print(scores)

        # Ensure scores are stored as numbers or strings (not lists)
        if isinstance(scores, list):
            scores = [str(score) for score in scores]  # Convert list of numbers to list of strings
        elif isinstance(scores, (float, int)):
            scores = str(scores)  # Convert single number to string

        metadata = {
            "categories": categories if isinstance(categories, (str, list)) else str(categories),
            "scores": scores if isinstance(scores, list) else [scores]  # Ensure it's a list of strings
        }
        
        doc = Document(page_content=item["review"], metadata=metadata)
        docs.append(doc)

    vectorstore.add_documents(docs)  # Store correctly formatted metadata
