# Python built-in libraries
import os
import pickle
import random
from collections import Counter

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data

DATA_PATH = "/root/capsule/data/sample/dominguez_conde_immune_tissue_two_donors.h5ad"

adata = anndata.read_h5ad(DATA_PATH)

def predict_cell_types(adata, model_path, save_dir, save_name):
    """
    Loads data and model to predict cell types using the Cell2Sentence library.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - model_path (str): Path or name of the pretrained Cell2Sentence model for cell type prediction.
    - save_dir (str): Directory to save processed CSData and results.
    - save_name (str): Name for saving the dataset and model.
    - seed (int): Random seed for reproducibility. Default is 1234.

    Returns:
    - List of predicted cell types for each cell in the dataset.
    """
    # Set random seed
    #random.seed(seed)
    #np.random.seed(seed)

    # Select relevant columns from adata.obs
    adata.obs = adata.obs[["cell_type", "tissue", "batch_condition", "organism", "sex"]]
    adata_obs_cols_to_keep = adata.obs.columns.tolist()

    # Create CSData object
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata, 
        #random_state=seed, 
        sentence_delimiter=' ',
        label_col_names=adata_obs_cols_to_keep
    )

    # Reduce dataset for faster processing (optional, use if needed)
    increm = arrow_ds.num_rows // 3
    arrow_ds = arrow_ds.select(range(0, arrow_ds.num_rows, increm))

    # Save CSData
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds, 
        vocabulary=vocabulary,
        save_dir=save_dir,
        save_name=save_name,
        dataset_backend="arrow"
    )

    # Load CSModel
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=save_dir,
        save_name=save_name
    )

    # Predict cell types
    predicted_cell_types = predict_cell_types_of_data(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=200
    )

    return predicted_cell_types


predict_cell_types(adata, model_path="vandijklab/C2S-Pythia-410m-cell-type-prediction",save_dir="/root/capsule/scratch/Model",save_name="C2S-Pythia-410m-cell-type-prediction")

import scanpy as sc
import numpy as np


def list_cell_types(file_path, return_as='list'):
    """
    Extracts unique cell types from a processed .h5ad file.

    Parameters:
    - file_path (str): Path to the .h5ad file.
    - return_as (str): Format to return the cell types ('list' or 'array').

    Returns:
    - List[str] or np.ndarray: List or array of unique cell types.
    """
    adata = sc.read_h5ad(file_path)
    
    if 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type'].unique().tolist()

        if return_as == 'array':
            return np.array(cell_types)  # Return as a NumPy array
        else:
            return cell_types  # Return as a list
    else:
        return []


file_path = "/root/capsule/data/test_1/dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0.h5ad"
list_cell_types(file_path)

import scanpy as sc
import matplotlib.pyplot as plt
import streamlit as st

def plotmap(file_path, color="cell_type", size=8, title="Human UMAP"):
    adata = sc.read_h5ad(file_path)
    
    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the UMAP and render it to the current figure
    sc.pl.umap(adata, color=color, size=size, title=title, show=False, ax=ax)
    
    # Display the plot in Streamlit
    st.pyplot(fig)


file_path = "/root/capsule/data/test_1/dominguez_conde_immune_tissue_two_donors_preprocessed_tutorial_0.h5ad"
plotmap(file_path=file_path)

import scanpy as sc
import pandas as pd

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
    #return top_genes_dict
    return top_genes_df

def get_top_genes_for_all_cell_types(adata, cell_type_col="cell_type", n_top_genes=5):
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
    top_genes_df : pd.DateFrame
        A dataframe (long format) containing n_top_genes differentially expressed genes alongside 
        adjusted p-values and log-fold changes for each cell type.
    """
    
    # Perform the ranking of genes using scanpy's built-in function
    sc.tl.rank_genes_groups(adata, groupby=cell_type_col, method='wilcoxon')

    # Extract the differential expression results
    rank_genes_groups = adata.uns['rank_genes_groups']
    # Extract the group (cell type) names
    groups = rank_genes_groups['names'].dtype.names
    # Initialize an empty list to store the results
    results = []
    # Loop through each cell type or cluster
    for group in groups:
        # Extract the gene names, p-values, and log fold changes for this group
        genes = rank_genes_groups['names'][group]
        pvals_adj = rank_genes_groups['pvals_adj'][group]
        logfoldchanges = rank_genes_groups['logfoldchanges'][group]
        # Append the results for this group to the results list
        for i in range(n_top_genes):
            results.append([group, genes[i], pvals_adj[i], logfoldchanges[i]])
    # Convert the results to a Pandas DataFrame
    top_genes_df = pd.DataFrame(results, columns=['cell_type', 'gene', 'pvals_adj', 'logfoldchange'])
    return top_genes_df

def get_average_expression_per_cell_type(adata, cell_type_col="cell_type"):
    """
    Returns the average expression level of each gene per cell type.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the gene expression data and cell annotations.
    cell_type_col : str
        The column in adata.obs that contains cell type annotations.

    Returns:
    --------
    avg_expression_df : pd.DataFrame
        A DataFrame where rows represent genes, columns represent cell types,
        and values are the average expression levels of the genes in each cell type.
    """

    # Extract the cell type annotations and create an array for faster operations
    cell_types = adata.obs[cell_type_col].values

    # Unique cell types in the data
    unique_cell_types = np.unique(cell_types)

    # Initialize a DataFrame to store average expression per cell type
    avg_expression_df = pd.DataFrame(index=adata.var_names, columns=unique_cell_types)

    # Loop through each cell type, subset data, and calculate the mean expression per gene
    for cell_type in unique_cell_types:
        # Subset AnnData object to include only cells of the current cell type
        subset = adata[cell_types == cell_type]
        
        # Calculate mean expression for each gene
        mean_expression = np.asarray(subset.X.mean(axis=0)).flatten()
        
        # Add the mean expression for this cell type to the DataFrame
        avg_expression_df[cell_type] = mean_expression

    return avg_expression_df

    # Python built-in libraries
import os
import pickle
import random
from collections import Counter

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


def Processing_function(file_name = 'dominguez_conde_immune_tissue_two_donors.h5ad'):
    
    # This function receives h5ad or csv files as an input and returns a cell2sentence 
    # object as an output. First, it filters the data and filter out the low quality 
    # cells and the genes that are expressed in a few cells. Then, it calculates the 
    # percentage mitochondrial genes. In the next step, it does normalization and 
    # principal component analysis (pca). To put cells into various clusters, it performs 
    # neighboring and then performs Uniform Manifold Approximation and Projection (UMAP).
    # The resulting UMAP plot and the cell2sentence object will be saved in and can be 
    # used for further analyses.
    
    
    
    # Fetching the data
    DATA_PATH = f"/root/capsule/data/uploaded_data/{file_name}"

    file_format = re.split(r'\.', file_name)
    print(file_format[-1])
    if file_format[-1] == 'csv':
        adata = anndata.read_csv(DATA_PATH)
    if file_format[-1] == 'h5ad':
        adata = anndata.read_h5ad(DATA_PATH)
    if file_format[-1] == 'h5':
        adata = sc.read_10x_h5(DATA_PATH)
        adata.var_names_make_unique()

        

    # Filtering the cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # annotate the group of mitochondrial genes as "mt"
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # adata = adata[adata.obs.n_genes_by_counts < 6000, :]
    # adata = adata[adata.obs.pct_counts_mt < 50, :].copy()

    # Count normalization
    sc.pp.normalize_total(adata)
    # Lop1p transformation with base 10 - base 10 is important for C2S transformation!
    sc.pp.log1p(adata, base=10)  
    
    # Run principal component analysis (pca)
    sc.tl.pca(adata)

    # Perform neighboring and clustering
    sc.pp.neighbors(adata)

    # Perform Uniform Manifold Approximation and Projection (UMAP)
    sc.tl.umap(adata)

    # Save the UMAP plot
    fig = plt.figure()
    sc.pl.umap(
        adata,
        color="cell_type",
        size=8,
        title="Human Immune Tissue UMAP",
        show= False
    )
    SAVE_PATH = f"/root/capsule/scratch/plots/umap_image.png"
    plt.savefig(SAVE_PATH, dpi = 300, format = 'png')
    plt.close(fig)

    # Save the cell2sentence object
    new_file_name = file_name[:-5]
    SAVE_PATH = f"/root/capsule/scratch/processed_data/{new_file_name}_processed.h5ad"
    adata.write_h5ad(SAVE_PATH)

#Processing_function()

import scanpy as sc
import pandas as pd
def get_top_genes_for_all_tissue_types(adata, tissue_type_col="tissue", n_top_genes=5):
    """
    Returns the top n marker genes for each tissue type using scanpy's rank_genes_groups function.    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing the expression data and cell annotations.
    cell_type_col : str
        The column in adata.obs that contains tissue type annotations.
    n_top_genes : int
        The number of top genes to return for each tissue type.    Returns:
    --------
    top_genes_dict : dict
        A dictionary where keys are tissue types, and values are DataFrames containing
        the top n genes and their scores for each cell type.
    """    # Perform the ranking of genes using scanpy's built-in function
    sc.tl.rank_genes_groups(adata, groupby=tissue_type_col, method='wilcoxon')    # Dictionary to store top genes for each cell type
    top_genes_dict = {}    # Get the unique cell types
    tissue_types = adata.obs[tissue_type_col].unique()    # Loop over each cell type and extract the top n ranked genes
    for tissue_type in tissue_types:
        # Extract the gene names and scores for the given cell type
        top_genes = adata.uns['rank_genes_groups']['names'][tissue_type][:n_top_genes]
        #p_vals = adata.uns['rank_genes_groups']['pvals_adj'][cell_type][:n_top_genes]
        #l_f_changes = adata.uns['rank_genes_groups']['logfoldchanges'][cell_type][:n_top_genes]        # Create a DataFrame to store the top genes and their scores
        #top_genes_df = pd.DataFrame({
        #    'gene': top_genes,
        #    'adjusted_p_values': p_vals,
        #    'logfold_changes': l_f_changes
        #})        # Add the top genes for this cell type to the dictionary
        #top_genes_dict[cell_type] = top_genes_df
        top_genes_dict[tissue_type] = top_genes
        top_genes_df = pd.DataFrame.from_dict(top_genes_dict,  orient="index")
    #return top_genes_dict
    return top_genes_df
5:53
get_top_genes_for_all_tissue_types(adata=adata)

import pandas as pd
# Extract the differential expression results
rank_genes_groups = adata.uns['rank_genes_groups']
# Extract the group (tissue) names
groups = rank_genes_groups['names'].dtype.names
# Initialize an empty list to store the results
results = []
# Loop through each cell type or cluster
for group in groups:
    # Extract the gene names, p-values, and log fold changes for this group
    genes = rank_genes_groups['names'][group]
    pvals_adj = rank_genes_groups['pvals_adj'][group]
    logfoldchanges = rank_genes_groups['logfoldchanges'][group]
    # Append the results for this group to the results list
    for i in range(len(genes)):
        results.append([group, genes[i], pvals_adj[i], logfoldchanges[i]])
# Convert the results to a Pandas DataFrame
df = pd.DataFrame(results, columns=['tissue', 'gene', 'pvals_adj', 'logfoldchange'])
# Display the DataFrame
print(df.head())







