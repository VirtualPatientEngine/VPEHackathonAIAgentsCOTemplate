# LLM-RAG for Biomedical Applications

Welcome to the **LLM-RAG for Biomedical Applications** project! This system combines powerful tools for analyzing single-cell RNA-seq data, simulating drug effects on eczema, and interpreting gene-cell type relationships. It's designed to help researchers and medical professionals explore gene expression datasets, test hypotheses about drug efficacy, and understand complex biological queries.

## Project Overview

This project integrates language models with retrieval-augmented generation (RAG), systems biology modeling, and gene-cell type interpretation to provide insightful answers to biomedical questions. It leverages the **Ollama** language model server for generating embeddings, processing natural language queries, and generating responses.

The system consists of three main components:

1. **RAG (Retrieval-Augmented Generation) Model:** For querying single-cell RNA-seq data.
2. **SysBio Model:** For simulating drug effects on eczema using the AD-QSP (Atopic Dermatitis Quantitative Systems Pharmacology) model.
3. **Cell2Sentence Model:** For interpreting user inquiries related to cell types and associated genes, providing professional responses.

All components are integrated into a user-friendly **Streamlit** application, making it easy to interact with complex biological data and models.

---
## Getting started - Cloning the GitHub Repo into a new capsule

### stage I: Clone the repo:
get your group GitHub repository address **<github_address>** (ex. https://github.com/VirtualPatientEngine/VPEHackathonAIAgentsCOTemplate)

1. Click on the **New Capsule** button on the top right corner.
2. Select: "**Copy from virtualPatientAgent**".
3. Paste the git repository address: **<github_address>**
4. Click **clone**
5. The capsule will be cloned within a few seconds.

### stage II: Create and add your personal access tokens

6. Follow GitHub instructions to generate `personal access token`: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
7. Then on Code Ocean, click on the 'account' icon on the bottom left side, and go to `credentials`.
8. Click on `⊕ Add credential` and choose `GitHub`, then add your username (in GitHub) and the token you have created.    

### stage III: Attach the data-assets:

9. In the capsule view, in the `data` folder in the files tree click **⚙️manage**
10. Attach the data-assets by clicking the plus sign (⊕):   

   a. ⊕ `collections: cellxgene census metadata 2024-04-24`.  
   b. ⊕ `ollama_models_09_2024`

Now, all the code in the capsule should be ready to run.


### Ollama important note:
The current setup maps the default Ollama models to `/data/`, making it easy to attach Ollama models as data assets. However, since data assets are immutable and cannot be changed, you’ll need to map the `./ollama` directory to `/scratch/.ollama` if you want to download or modify different models, as `/scratch/` is writable.

```bash
# Create the source directory if it doesn't exist
mkdir -p /scratch/.ollama

# delete the existing symbolic link link
rm /root/.ollama

# Create the new symbolic link (each write to /root/.ollama will be directed to /scratch/.ollama)
ln -s  /scratch/.ollama /root/.ollama

# copy the key:
cp /data/.ollama/id_ed25519 /scratch/.ollama/id_ed25519

```
... Now you can easily pull any ollama model with `ollama pull <model>`
---


## Key Components and Files

### 1. Streamlit Application (`streamlit_app.py`)

**Purpose:**

This is the main entry point of the application. It provides an intuitive web interface for users to interact with the RAG model, the SysBio model, and the Cell2Sentence model. It sets up three main tabs:

- **Ask about Single-Cell RNA-seq Data:** Users can input questions related to gene expression, and the app searches for relevant datasets, displaying them alongside summaries generated by the language model.
- **Talk with SysBio Model:** Users can interact with the AD-QSP model by describing potential drug effects, and the app simulates their impact on eczema severity, providing predicted EASI scores and severity levels.
- **Ask about Cells:** Users can input questions about cell types or lists of genes, and the app generates clear, professional responses identifying the relevant cell types or expected genes based on the provided information.

**Key Features:**

- **Model Selection:** Allows users to select from different language models available in Ollama.
- **Real-time Interaction:** Provides immediate responses to user inputs, leveraging streaming capabilities.
- **Integration with Ollama:** Uses Ollama for natural language processing, including prompt-based interactions, streaming responses, embedding generation, and RAG.
- **Cell2Sentence Integration:** Enables users to ask questions about cell types and genes, facilitating accurate interpretation of complex gene-cell type relationships.

**How to Use:**

Run the Streamlit app and navigate between the tabs. Input your questions or drug descriptions, and the app will provide responses based on the integrated models.

- **Ask about Single-Cell RNA-seq Data Tab:** Input questions related to gene expression to find relevant datasets.
- **Talk with SysBio Model Tab:** Describe potential drug effects to simulate their impact on eczema severity.
- **Ask about Cells Tab:** Enter questions about cell types or gene lists to receive professional interpretations.

---

### 2. Cell2Sentence Module (`cell2Sentence.py`)

**Purpose:**

This module processes user inquiries related to cell types and associated genes by integrating a pre-trained causal language model with the **Ollama** API. It intelligently determines whether a user's query specifies cell types or gene lists, then generates relevant prompts to extract expected genes or predict cell types accordingly. It facilitates accurate interpretation and communication of complex gene-cell type relationships by automating the response generation process.

**Key Functions:**

- `cell2Sentence(query)`: Generates a response based on the query using a pre-trained language model.
- `question_to_cell_type(query)`: Determines if the user's query specifies cell types or a list of genes and generates the appropriate prompt to extract expected genes or predict cell types.

**How it Uses Ollama:**

- **Response Refinement:** Uses the `ollama_response` function from `ollama_utils.py` to refine the outputs from the pre-trained language model into concise, professional responses.

**How to Use:**

Use `question_to_cell_type(query)` to process user questions about cell types or gene lists. For example:

```python
from cell2Sentence import question_to_cell_type

result = question_to_cell_type("What cell types are associated with genes MT-CO3 and MT-ATP6?")
print(result)
```

**Integration with the App:**

- In the Streamlit app, the **"Ask about Cells"** tab allows users to input questions related to cell types or genes. The app uses `question_to_cell_type` to generate and display professional responses.

---

### 3. Embedding Utilities (`embedding_utils.py`)

**Purpose:**

This module provides utilities for creating and managing a ChromaDB collection of embedded text documents. It leverages embeddings generated by Ollama to add documents to the database, enabling efficient semantic search and retrieval.

**Key Functions:**

- `document_exists(collection, doc)`: Checks if a document already exists in the collection.
- `add_document(collection, doc, doc_id)`: Adds a document and its embedding to the collection if it doesn't already exist.
- `process_documents(client, docs, collection_name)`: Processes a list of documents, embedding them, and adding them to a ChromaDB collection.
- **Main Block:** Demonstrates how to load documents, create a ChromaDB collection, process documents, and query the collection.

**How it Uses Ollama:**

- **Embedding Generation:** Uses the `get_embedding` function from `ollama_utils.py` to generate embeddings for documents using Ollama's embedding models.

**How to Use:**

Use these functions to build and manage your embedding database. The main block shows an example of processing documents and querying the collection.

---

### 4. AD-QSP Tools (`AD_QSP_tools.py`)

**Purpose:**

This module contains tools for simulating drug effects on eczema severity using the AD-QSP model from Miyano et al. It defines the mathematical model, loads patient parameters, and provides functions to simulate patient responses to drug effects, compute EASI scores, and assess eczema severity.

**Key Functions:**

- `load_parameters(mu_file, sigma_file)`: Loads model parameters from CSV files.
- `simulate_patient(x, de)`: Simulates a patient's response to drug effects.
- `test_drug_efficacy(drug_effects, n_patients)`: Simulates drug effects across virtual patients.
- `get_easi_severity(easi_score)`: Determines eczema severity based on the EASI score.
- `question_examples`: Provides example questions to guide users in formulating queries.

**How it Integrates with the App:**

In the Streamlit app, when a user describes a drug's effects, these functions are used to simulate the impact on eczema severity and provide predictions.

**How to Use:**

Define your drug effects as a dictionary and use `test_drug_efficacy()` to simulate and get results. For example:

```python
drug_effects = {"IL-4": -0.5, "Th1": 0.1}
results = test_drug_efficacy(drug_effects)
```

---

### 5. Gene Expression Datasets RAG (`gene_expression_datasets_rag.py`)

**Purpose:**

This module provides helper functions to initialize and use the Retrieval-Augmented Generation (RAG) system within the Streamlit app. It handles loading datasets and the embedding database into the app's session state.

**Key Functions:**

- `load_rag_system()`: Initializes the RAG system by loading the necessary datasets and embeddings.
- `search_by_question(question)`: Searches the ChromaDB collection for relevant documents based on a user's question, using embeddings.

**How it Uses Ollama:**

- **Embedding Generation:** Calls `get_embedding` from `ollama_utils.py` to generate embeddings for the user's question, enabling semantic search.

**How to Use:**

Call `load_rag_system()` at the start of your app to initialize, and use `search_by_question(question)` to retrieve relevant datasets based on a query.

---

### 6. Query Utilities (`query_utils.py`)

**Purpose:**

This module contains utility functions for processing and formatting query results, especially for display in the app.

**Key Functions:**

- `convert_list_columns_to_str(df)`: Converts columns in a DataFrame that contain lists of dictionaries into strings for better readability.
- `from_collection_datasets_to_markdown(datasets)`: Generates a markdown table from datasets, suitable for display in the Streamlit app.
- `extract_dictionary_from_response(result)`: Extracts a dictionary from a string response, useful for parsing language model outputs into usable data structures.

**How to Use:**

Use these functions to prepare data for display or further processing within the app.

---

### 7. Ollama Utilities (`ollama_utils.py`)

**Purpose:**

This module provides utility functions for interacting with the **Ollama** language model server. It ensures that the Ollama server is running and provides functions for generating responses, streaming responses, and generating embeddings.

**Key Functions:**

- `restart_ollama_server()`: Ensures the Ollama server is running, starting it if necessary.
- `ollama_response(message, model, role, temperature)`: Generates a response from the language model for a given prompt.
- `ollama_stream(message, model, role, temperature)`: Streams a response from the language model in real-time, useful for updating the UI as the response is generated.
- `get_embedding(input, model)`: Generates embeddings for input text using Ollama's embedding models.

**How to Use:**

- **Prompt-based Interaction:** Use `ollama_response()`
to get responses from the language model based on prompts.
- **Streaming Responses:** Use `ollama_stream()` to stream responses in real-time.
- **Embedding Generation:** Use `get_embedding()` to generate embeddings for text, which can then be used in semantic search or RAG systems.

**Integration with the App:**

- The Streamlit app uses these functions to interact with Ollama for generating summaries, processing user queries, and generating embeddings for the RAG system.
- The `cell2Sentence.py` module also uses `ollama_response()` to refine model outputs.

---

### 8. CellxGene Utilities (`cellxgene_utils.py`)

**Purpose:**

This module includes utilities related to CellxGene data, specifically importing the `cellxgene_census` package. It contains code demonstrating how to download datasets from CellxGene.

**Note:**

- Currently, the code in this module is commented out and serves as a reference for how to work with CellxGene data.

**How to Use:**

Refer to this module when you need to download or interact with CellxGene datasets using the `cellxgene_census` package.

---

## Emphasizing the Use of Ollama

The **Ollama** language model server is a central component of this project, used extensively for:

- **Prompt-based Interactions:** Generating natural language responses to user queries.
- **Streaming Responses:** Providing real-time feedback to the user as the model generates responses.
- **Embedding Generation:** Creating vector embeddings of text documents and user queries for semantic search and RAG.
- **RAG (Retrieval-Augmented Generation):** Enhancing the language model's responses by retrieving relevant documents from a database (ChromaDB) and incorporating that information into the generated responses.
- **Response Refinement:** Refining outputs from pre-trained models into concise, professional responses.

**How Ollama is Used in the Project:**

- **In `streamlit_app.py`:**
  - Uses `ollama_stream()` to generate and display language model responses in real-time as the user interacts with the app.
  - Calls `ollama_response()` to process and extract parameters from user input for the SysBio model.
  - Integrates with the **Cell2Sentence** model to process user queries about cell types and genes.
- **In `embedding_utils.py` and `gene_expression_datasets_rag.py`:**
  - Uses `get_embedding()` to generate embeddings for documents and user queries, facilitating efficient semantic search in the RAG system.
- **In `cell2Sentence.py`:**
  - Uses `ollama_response()` to refine the outputs from the pre-trained language model into concise, professional responses.
- **In `ollama_utils.py`:**
  - Provides the utility functions that interface with Ollama's API, ensuring seamless integration with the rest of the application.

**Benefits of Using Ollama:**

- **Flexibility:** Supports multiple language models, allowing for experimentation and optimization.
- **Real-time Interaction:** Streaming capabilities enhance user experience by providing immediate feedback.
- **Powerful Embeddings:** High-quality embeddings improve the effectiveness of semantic search and RAG systems.
- **Professional Responses:** Refines model outputs to generate clear and professional responses to complex queries.

**Example Usage of Ollama Functions:**

- **Generating a Response:**

  ```python
  from ollama_utils import ollama_response

  prompt = "Explain the significance of IL-4 in eczema."
  response = ollama_response(prompt, model="llama3.1")
  print(response)
  ```

- **Streaming a Response:**

  ```python
  from ollama_utils import ollama_stream

  prompt = "Summarize the following document..."
  for chunk in ollama_stream(prompt, model="llama3.1"):
      print(chunk, end="", flush=True)
  ```

- **Generating Embeddings:**

  ```python
  from ollama_utils import get_embedding

  text = "Gene expression analysis of skin samples."
  embedding = get_embedding(text, model="nomic-embed-text")
  ```

## How to Use the Project

1. **Set Up the Environment:**

   - Ensure that **Ollama**, **ChromaDB**, and **Hugging Face Transformers** are installed and configured properly.
   - Make sure the necessary data files are available (e.g., CellxGene collections metadata, model parameter CSV files).

2. **Run the Streamlit App:**

   - Execute `streamlit_app.py` to start the web application.
   - The app will handle initializing the RAG system and ensuring Ollama is running.

3. **Explore the App Functionality:**

   - **Ask about Single-Cell RNA-seq Data:**
     - Navigate to the **"Ask about Single-Cell RNA-seq Data"** tab.
     - Enter a question or topic related to gene expression.
     - The app will search for relevant datasets, display them, and provide summaries generated by the language model.
   - **Simulate Drug Effects on Eczema:**
     - Navigate to the **"Talk with SysBio Model"** tab.
     - Describe a drug's effects on specific biological parameters (e.g., "This drug lowers IL-4 by 50% and slightly increases Th1.").
     - The app will extract the parameters, run simulations using the AD-QSP model, and display the predicted EASI scores and severity levels.
   - **Ask about Cells:**
     - Navigate to the **"Ask about Cells"** tab.
     - Enter a question about cell types or a list of genes (e.g., "What cell types are associated with genes MT-CO3 and MT-ATP6?").
     - The app will generate a professional response identifying relevant cell types or expected genes based on your input.

4. **Understand and Modify the Code:**

   - Explore the modules and understand how each component works.
   - Use the provided functions to extend or customize the application as needed.
   - The code is organized to separate concerns, making it easier to maintain and understand.

---

## Expanded Use Cases: Machine Learning and Systems Biology

The project has now been extended with several powerful machine learning, vector search, and systems biology tools. These new features enable the project to support a wider range of tasks such as semantic search, dynamic simulations of biological models, and the construction of knowledge graphs. Below are the details for each new example, organized by file name:

### File: `ollama_faiss_example.py`

- **Purpose**: This example demonstrates how to create a **semantic search system** using FAISS and the `nomic-embed-text` model from Ollama.
- **Key Features**: 
  - Document embeddings are created using `OllamaEmbeddings` and stored in a FAISS index for fast similarity-based retrieval.
  - The FAISS index is used to retrieve the most similar document based on a query.
- **Use Case**: Ideal for situations where fast and scalable document retrieval is needed, such as finding relevant papers or documents in large datasets.
- **How to Use**: Use the embeddings generated from textual data to create an index for querying. The code is designed to store and retrieve documents based on semantic similarity.

### File: `basico_example.py`

- **Purpose**: This script replicates the **Teusink2000 glycolysis model** using the `basico` library to simulate steady-state fluxes and metabolite concentrations.
- **Key Features**: 
  - The model replicates the results from Teusink et al. (2000) with slight adjustments for ATP species and kinetic equations.
  - Time-course simulations allow users to explore dynamic changes in glycolysis over time.
- **Use Case**: Useful for systems biology researchers interested in enzyme kinetics and metabolic pathways.
- **How to Use**: Load the Teusink2000 glycolysis model, run a simulation, and analyze the results, such as metabolite concentrations over time.

### File: `langchain_ollama_example.py`

- **Purpose**: Demonstrates **caching with LangChain** to optimize repeated queries using the `OllamaLLM` model.
- **Key Features**: 
  - Integrates an SQLite cache to store responses, improving performance on repeated or similar queries.
  - Uses a prompt template to interact with the Ollama model for natural language processing tasks.
- **Use Case**: This setup is perfect for applications where repeated queries are common, reducing inference time and improving overall system efficiency.
- **How to Use**: Implement the SQLite cache and set up the `OllamaLLM` model to optimize repeated requests and enhance response times.

### File: `langchain_torch_geometric_example.py`

- **Purpose**: Converts unstructured text into a **knowledge graph** using `LLMGraphTransformer` and the Ollama LLM.
- **Key Features**: 
  - Extracts entities and relationships from text and constructs a graph representation using NetworkX.
  - The graph can be saved for visualization or further analysis.
- **Use Case**: This example is useful for building knowledge graphs from textual data, such as scientific literature or historical records.
- **How to Use**: Use the `LLMGraphTransformer` to extract entities and relationships, then construct and visualize the graph with NetworkX.

### File: `example_torch_geometric.py`

- **Purpose**: This example demonstrates using **Graph Neural Networks (GNNs)** for **regression tasks in drug discovery**, using the PyTorch Geometric library.
- **Key Features**: 
  - A GCN model is trained on the MoleculeNet “lipo” dataset to predict molecular properties.
  - The script includes functions for dataset loading, model training, and evaluation.
- **Use Case**: This is ideal for tasks in cheminformatics or drug discovery where molecular properties need to be predicted based on graph structures.
- **How to Use**: Load the MoleculeNet dataset, train a GCN model, and evaluate its performance on regression tasks such as predicting molecular binding affinity.

### Purpose of the New Systems

The newly added examples significantly expand the scope of this project by incorporating advanced machine learning, systems biology, and knowledge graph construction capabilities. These additions allow the project to:

1. **Enable Scalable Semantic Search**: By using FAISS and `OllamaEmbeddings`, users can create efficient document retrieval systems.
  
2. **Simulate Complex Biological Systems**: The inclusion of the Teusink2000 glycolysis model enables dynamic simulations of biochemical pathways, providing insights into metabolic fluxes and enzyme kinetics.

3. **Build Knowledge Graphs from Text**: Using `LLMGraphTransformer`, unstructured text can be transformed into structured knowledge graphs, useful for relationship extraction and data visualization.

4. **Apply Machine Learning in Drug Discovery**: The GCN model example shows how to apply graph neural networks for molecular property prediction, crucial for drug discovery and cheminformatics research.

By integrating these machine learning, simulation, and NLP tools, the project now provides a comprehensive platform for cutting-edge research across multiple domains. Users can seamlessly switch between NLP tasks, machine learning model training, and biological simulations, all within a unified environment.