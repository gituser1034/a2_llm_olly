from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Will need to figure out how to get data from pdf later
df = pd.read_csv("rag_doc.csv")
# Error with finding this
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

# Folder where storing database
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Checking if db already exists
# Adding langchain documents and ids to lists to be built into db later
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Database entry - like a row in the database
        document = Document(
            # What were vectorizing and looking up - querying
            # Page content like the text, metadata = labels, extra info about text
            # Only need models opinion based on review, date it was given don't  matter 
            page_content=row["Title"] + " " + row["Review"],
            # Not querying by metadata, extra data, not embedded
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Creating empty datastore/db
vector_store = Chroma(
    collection_name = "restaurant_reviews",
    persist_directory = db_location,
    embedding_function=embeddings
)

# Adding data based on previous ids and documents to the datastore
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# looking up documents - here looking up 3 relevant review
# We can later pass these into prompt for llm
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

