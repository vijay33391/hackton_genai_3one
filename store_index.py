import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.helper import extract_data, filter_to_minimal_docs, split_text, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore



# api
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
gemini_api_key=os.environ.get('gemini_api_key')
''''
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMMINI_API_KEY"] =gemini_api_key'''

# Extract data from PDF files in the specified folder
extracted_data=extract_data(folder_path="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=split_text(filter_data)

pinecone_api_key = PINECONE_API_KEY

embedding = download_embeddings()
# create the index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "insurence-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Use a region available in your Pinecone project
    )

index = pc.Index(index_name)

# push the data to the index

# push embeddings to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)