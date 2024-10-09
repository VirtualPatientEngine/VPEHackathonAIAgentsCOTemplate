"""
This code demonstrates how to integrate caching in LangChain using an Ollama model
and SQLite. The `SQLiteCache` stores previous model responses to optimize future 
queries. A prompt template is defined to format questions, which are passed to the 
Ollama model (`llama3.1`) for inference. The results are cached to improve 
performance on repeated or similar queries. The cache is stored in a scratch 
directory defined relative to the current script's path.
"""

from langchain_ollama import OllamaLLM
from langchain_community.cache import SQLiteCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from ollama_utils import restart_ollama_server
from pathlib import Path

scratch_path = Path(__file__).resolve().parent.parent / "scratch"
restart_ollama_server() # necessary in a capsule ! - to start the ollama server first!
# Setup caching
langchain_cache = SQLiteCache(database_path=f"{scratch_path}/langchain_cache.db")
model = OllamaLLM(model="llama3.1", cache=langchain_cache)
template = """Question: {question} Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)


chain = prompt | model
result = chain.invoke({"question": "What is LangChain?"})
print(result)