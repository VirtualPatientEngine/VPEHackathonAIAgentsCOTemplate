# code/initialize_rag.py
# helper function for the streamlit_app that demonstrates how to use stramlit sesiosn state to store the dataset and embedding database

import streamlit as st
from pathlib import Path
import pandas as pd
import chromadb
from ollama_utils import restart_ollama_server
from embedding_utils import get_embedding


# --- Section: Load the dataset and embedding database ---
data_dir = "/data" if Path("/data/collections").exists() else Path(__file__).parent.parent / "data"
scratch_dir = "/scratch" if Path("/scratch").exists() else Path(__file__).parent.parent / "scratch"
MAX_DATASETS = 2

def load_rag_system():
    # global session to check if the ollama server has been started...
    if "ollama_started" not in st.session_state:
        restart_ollama_server()
        st.session_state.ollama_started = True
    if "dataset" not in st.session_state:
        st.session_state.dataset = pd.read_parquet(f"{data_dir}/collections/cellxgene_collections_metadata.parquet")
        print("Dataset loaded.")
    if "collection" not in st.session_state:
        st.session_state.client = chromadb.PersistentClient(path=f"{scratch_dir}/cellxgene_collections_chromadb")
        st.session_state.collection = st.session_state.client.get_collection(name="descriptions")
        print("Embedding database loaded.")


def search_by_question(question: str):

    results = st.session_state.collection.query(
        query_embeddings=[get_embedding(question)[0]], n_results=5, include=["documents", "distances"]
    )
    ids = results["ids"][0][0:MAX_DATASETS]
    docs = results["documents"][0][0:MAX_DATASETS]
    return ids, docs
