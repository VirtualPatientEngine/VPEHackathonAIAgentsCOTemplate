"""
Instructions:
1. Obtain a processed example dataset e.g., using https://github.com/vandijklab/cell2sentence/blob/master/tutorials/c2s_tutorial_0_data_preparation.ipynb
2. Update the path to the dataset i.e., file_path = "/root/capsule/data/sample/dominguez_conde_immune_tissue_two_donors.h5ad"
3. Run
"""

import streamlit as st
import scanpy as sc
import langchain
from langchain.llms import Ollama  # Ensure you have access to Ollama's Llama 3.1
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PIL import Image
import scanpy as sc
import matplotlib.pyplot as plt

import scanpy as sc
import numpy as np
import pandas as pd

import sys
sys.path.append('/root/capsule/code/demo')
from ollama_utils import ollama_stream
from gene_expression_datasets_rag import load_rag_system

st.set_page_config(page_title="LLM-RAG-for-biology-examples", page_icon="🌟", layout="wide")
load_rag_system()

def plotmap(adata, color="cell_type", size=8, title="Human UMAP"):

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    # Count normalization
    sc.pp.normalize_total(adata)
    # Lop1p transformation with base 10 - base 10 is important for C2S transformation!
    sc.pp.log1p(adata, base=10) 
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)

    sc.tl.umap(adata)
    
    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the UMAP and render it to the current figure
    sc.pl.umap(adata, color=color, size=size, title=title, show=False, ax=ax)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def llama_answer(dataset):
    prompt_template = PromptTemplate(
            input_variables=["query", "dataset_info"],
            template="""
            You are an expert in biological datasets. Given the following dataset:
            {dataset}

            Answer the following query: {query}
            """,
        )

    # Create a chain with the LLM and prompt template
    chain = LLMChain(llm=llama, prompt=prompt_template)

        # Show a spinner while waiting for the LLM response
    with st.spinner("Processing your query..."):
        # Run the query
        response = chain.run({"query": user_query, "dataset": dataset})

    # Display the result from Ollama Llama 3.1
    st.write("Response from Llama 3.1:")
    st.write(response)


def list_cell_types(adata, return_as='list'):
    """
    Extracts unique cell types from a processed .h5ad file.

    Parameters:
    - file_path (str): Path to the .h5ad file.
    - return_as (str): Format to return the cell types ('list' or 'array').

    Returns:
    - List[str] or np.ndarray: List or array of unique cell types.
    """
    
    if 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type'].unique().tolist()


        
        llama_answer(str(cell_types )) # Return as a list
    else:
        st.write("something went wrong")

def get_top_genes_for_all_cell_types(adata, cell_type_col="cell_type", n_top_genes=100):
    """
    Returns the top n marker genes for each cell type using scanpy's rank_genes_groups function.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the expression data and cell annotations.
    cell_type_col : str
        The column in adata.obs that contains cell type annotations.
    n_top_genes : int
        The number of top genes to return for each cell type.

    Returns:
    --------
    top_genes_dict : dict
        A dictionary where keys are cell types, and values are DataFrames containing
        the top n genes and their scores for each cell type.
    """
    
    # Perform the ranking of genes using scanpy's built-in function
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    # Count normalization
    sc.pp.normalize_total(adata)
    # Lop1p transformation with base 10 - base 10 is important for C2S transformation!
    sc.pp.log1p(adata, base=10)

    sc.tl.rank_genes_groups(adata, groupby=cell_type_col, method='wilcoxon')

    # Dictionary to store top genes for each cell type
    top_genes_dict = {}

    # Get the unique cell types
    cell_types = adata.obs[cell_type_col].unique()

    # Loop over each cell type and extract the top n ranked genes
    for cell_type in cell_types:
        # Extract the gene names and scores for the given cell type
        top_genes = adata.uns['rank_genes_groups']['names'][cell_type][:n_top_genes]
        #p_vals = adata.uns['rank_genes_groups']['pvals_adj'][cell_type][:n_top_genes]
        #l_f_changes = adata.uns['rank_genes_groups']['logfoldchanges'][cell_type][:n_top_genes]

        # Create a DataFrame to store the top genes and their scores
        #top_genes_df = pd.DataFrame({
        #    'gene': top_genes,
        #    'adjusted_p_values': p_vals,
        #    'logfold_changes': l_f_changes
        #})

        # Add the top genes for this cell type to the dictionary
        #top_genes_dict[cell_type] = top_genes_df
        top_genes_dict[cell_type] = top_genes

    top_genes_df = pd.DataFrame.from_dict(top_genes_dict,  orient="index")
    llama_answer(top_genes_df.to_string())
    

# Set up the Ollama Llama 3.1 model
llama = Ollama(model="llama3.1")

# Streamlit app to query the h5ad file with the LLM
st.title("Talk2Cell with Ollama Llama 3.1")


# Hardcoded file path to the h5ad dataset
file_path = "/root/capsule/data/sample/dominguez_conde_immune_tissue_two_donors.h5ad"

# Load the AnnData dataset using Scanpy
st.write("Loading AnnData dataset...")
adata = sc.read_h5ad(file_path)

st.write("AnnData loaded successfully!")
st.write(f"Dataset contains {adata.n_obs} observations and {adata.n_vars} variables.")

# Display an input box to query the dataset
user_query = st.text_input("Ask a question about the dataset:")

if user_query:
    # Create a prompt template to query Ollama Llama 3.1
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
        You are an expert at routing questions to appropriate predefined function. 
        ! IMPORTANT !: Do not add any additional information, comments, or code.
        route the following user query : {query}
        Identify if the user is asking for the top n marker genes for each cell type or 
        the query is asking to Extracting unique cell types or the query is asking to ploting a graph.

        if the user is asking for the top n marker genes for each cell type
         return a specific prompt wrapped in a <prompt> tag as follows:
        <prompt>"get_top_genes"</prompt>
        IF the query is asking to Extracting unique cell types
        <prompt>"list_cell_types"</prompt>
        IF the query is asking to ploting a graph
        <prompt>"plotmap"</prompt>
        ! IMPORTANT !: Only return the <prompt> tag. Do not add any additional information, comments, or code.
        """,
    )

    # Get a basic summary of the dataset
    dataset_info = f"The dataset has {adata.n_obs} observations (cells) and {adata.n_vars} variables (genes)."

    # Create a chain with the LLM and prompt template
    chain = LLMChain(llm=llama, prompt=prompt_template)

    # Run the query
    with st.spinner("Processing your query..."):
        response = chain.run({"query": user_query})
    
    print("LLAMA RESPONSE 1",response)

    # Decision Node based on user query
    def decide_function(result):
        if "genes" in result:
            get_top_genes_for_all_cell_types(adata)
        elif "cell types" in result:
            list_cell_types(adata)
        elif "plot" in result:
            # Load an image from a file
            plotmap(adata)

    decision = decide_function(response)

    # Display the result from Ollama Llama 3.1
    st.write("Response from Llama 3.1:")
    # st.write(response)

    # Optionally show the dataset structure (e.g., first few rows)
    if st.checkbox("Show dataset structure (first 5 rows)?"):
        st.write(adata.obs.head())
        st.write(adata.var.head())

else:
    st.write("")
