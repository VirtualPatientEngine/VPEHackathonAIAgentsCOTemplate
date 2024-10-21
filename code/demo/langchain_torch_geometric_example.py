"""
This script demonstrates how to use the Ollama LLM to convert unstructured 
text into a knowledge graph using the `LLMGraphTransformer`. It first sets 
up the necessary imports and starts the Ollama LLM server, using SQLite 
caching for efficient repeated requests. The `LLMGraphTransformer` processes 
the provided text, extracting entities and relationships and converting them 
into a graph document. The script then builds a NetworkX graph by adding 
nodes and edges based on the extracted data. Finally, the graph is optionally 
saved as a GML file for further visualization or analysis.
"""


import networkx as nx
from langchain_ollama import OllamaLLM
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.cache import SQLiteCache
from ollama_utils import restart_ollama_server
from pathlib import Path

# Setup caching and Ollama LLM server
scratch_path = Path(__file__).resolve().parent.parent / "scratch"
restart_ollama_server()  # Start the Ollama server
langchain_cache = SQLiteCache(database_path=f"{scratch_path}/langchain_cache.db")

# Instantiate the Ollama model
llm = OllamaLLM(model="llama3.1", cache=langchain_cache)

# Initialize the LLMGraphTransformer with Ollama LLM
llm_transformer = LLMGraphTransformer(llm=llm)

# Example text for extraction
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

# Convert the text into a graph document using Ollama
documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Initialize an empty NetworkX graph
G = nx.Graph()

# Add nodes and relationships to the NetworkX graph
for node in graph_documents[0].nodes:
    G.add_node(node.id, type=node.type)

for relationship in graph_documents[0].relationships:
    G.add_edge(relationship.source.id, relationship.target.id, type=relationship.type)

# Output the nodes and edges
print("Nodes in the graph:", G.nodes(data=True))
print("Edges in the graph:", G.edges(data=True))

# Optionally, save the graph to a file for visualization
nx.write_gml(G, f"{scratch_path}/knowledge_graph.gml")