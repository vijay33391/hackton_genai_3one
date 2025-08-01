import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings


def extract_data(folder_path):
    """
    Extracts text from all PDF files in the specified folder.
    
    Args:
        folder_path (str): The path to the folder containing PDF files.
        
    Returns:
        list: A list of text extracted from each PDF file.
    """
    # Load all PDF files from the directory
    loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split the text into smaller chunks
def split_text(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits the text in each Document into smaller chunks.
    
    Args:
        documents (List[Document]): List of Document objects to be split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        List[Document]: List of Document objects with split text.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# Download embeddings from HuggingFace model
def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    see: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    size: 384 dimensions, 6 layers, 12 heads, 22M parameters
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings