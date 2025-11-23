from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd

loader = PyPDFLoader("AstroSeasons.pdf")
# Gives long list of all text from pdf including /n for newlines
# Has metadata saying producer - microsoft and page number, page_content stores text per page
pdf_docs = loader.load()
print(pdf_docs)

# Splitting data into smaller chunks for easier embedding
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)

# List of langchain documents
chunks = splitter.split_documents(pdf_docs)
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

# Folder where storing database
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Creating empty datastore/db
vector_store = Chroma(
    collection_name = "astro_notes",
    persist_directory = db_location,
    embedding_function=embeddings
)

# Adding data based on previous ids and documents to the datastore
if add_documents:
    vector_store.add_documents(chunks)

# looking up documents - here looking up 3 most relevant data chunks
# This gets passed into prompt for llm later
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)