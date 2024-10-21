"""
This code demonstrates how to use FAISS with the `nomic-embed-text` model 
via `OllamaEmbeddings` to create document embeddings. It generates document 
vectors, stores them in a FAISS index, and retrieves the most similar document 
based on a query. The embedding object is passed directly to FAISS, ensuring 
compatibility with future updates. The code shows how to handle document 
embedding, vector storage, and retrieval, providing a basic example of a 
semantic search system.
"""


from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from ollama_utils import restart_ollama_server
from uuid import uuid4


# Restart the Ollama server, necessary in a capsule
restart_ollama_server()

# Use the nomic-embed-text model with OllamaEmbeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

# Sample texts and metadata for documents
document_data = [
    ("I had chocolate chip pancakes and scrambled eggs for breakfast this morning.", "tweet"),
    ("The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.", "news"),
    ("Building an exciting new project with LangChain - come check it out!", "tweet"),
    ("Robbers broke into the city bank and stole $1 million in cash.", "news"),
    ("Wow! That was an amazing movie. I can't wait to see it again.", "tweet"),
    ("Is the new iPhone worth the price? Read this review to find out.", "website"),
    ("The top 10 soccer players in the world right now.", "website"),
    ("LangGraph is the best framework for building stateful, agentic applications!", "tweet"),
    ("The stock market is down 500 points today due to fears of a recession.", "news"),
    ("I have a bad feeling I am going to get deleted :(", "tweet"),
]

# Create a list of Document objects
documents = [
    Document(page_content=content, metadata={"source": source}) 
    for content, source in document_data
]

# Create FAISS index based on embedding dimensions
sample_embedding = embeddings.embed_query("hello world")
index = faiss.IndexFlatL2(len(sample_embedding))

# Create the FAISS vector store with the Embeddings object
vector_store = FAISS(
    embedding_function=embeddings,  # Pass the embedding object, not the function
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Generate unique UUIDs for the documents and add them to the vector store
uuids = [str(uuid4()) for _ in documents]
vector_store.add_documents(documents=documents, ids=uuids)

# Use the vectorstore as a retriever
retriever = vector_store.as_retriever()

# Retrieve the most similar document based on a query
retrieved_documents = retriever.invoke("What is LangChain?")

# Output the retrieved document's content
print(retrieved_documents[0].page_content)