import streamlit as st
import pandas as pd
import ollama
from ollama_utils import ollama_stream, ollama_response
from query_utils import from_collection_datasets_to_markdown, extract_dictionary_from_response
from gene_expression_datasets_rag import load_rag_system, search_by_question
from AD_QSP_tools import test_drug_efficacy, get_easi_severity, question_examples
from cell2Sentence import question_to_cell_type
import networkx as nx
import matplotlib.pyplot as plt
from stark_qa.tools.graph import k_hop_subgraph
from stark_qa import load_skb
from langchain_community.graphs import NetworkxEntityGraph
import gravis as gv
import streamlit.components.v1 as components


st.set_page_config(page_title="LLM-RAG-for-biology-examples", page_icon="🌟", layout="wide")
load_rag_system()


# ----------------------------------------------------------
# --- Section: Streamlit App ---

default_model = "llama3.1:latest"
models = [m["name"] for m in ollama.list()["models"] if not "embed" in m["name"]]
st.title("Ask me bio-medical question")
selected_model = st.selectbox("Select a model:", models, index=models.index(default_model))

if "skb" not in st.session_state:
    st.session_state.skb = load_skb("prime", download_processed=True, root="/scratch")

qa_dict = {"What potential health issue could serve as a link between immune-mediated acquired neuromuscular junction diseases and other neuromuscular junction diseases?":37776,
                "Are there any diseases associated with genetic hair shaft abnormalities that can be passed from parent to child?":28034, 
                "What disease shares a hierarchical connection as both a subtype and supertype with malignant pericardial mesothelioma and also has a related hierarchical link to epicardium cancer?":36973, 
                "What disease is linked to the HTR1A gene/protein and causes recurring fevers during the luteal phase of the menstrual cycle in women?":29620,
                "Which pathway is subordinate to 'Metabolic disorders of biological oxidation enzymes' and involved in adrenal cortical hormone biosynthesis?":128582,
                "What is the inherited dental disorder characterized by irregularities in both baby and adult teeth, with a birth incidence of 1 in 6000 to 1 in 8000?":39179
                }
st.title("Pick a question")
selected_q = st.selectbox("Select a query:", options=list(qa_dict.keys()))

@st.fragment
def do_qa():
    if question := st.chat_input("Ask me a question about this graph:"):
        mygraph = NetworkxEntityGraph(G)

        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            prompt = "You are a helpful assistant, there is a user input question which is followed by UserQ: and then a graph for your information in the format of tuples where the last entry of each 3 entry tuple is the relationship to another node, do not display the graph of relevant medical information just report useful information from it to the user in response to the query with a focus on information present in the provided graph, also at every question rank the nodes (only 1-3) corresponding to which most closely matches the users question ending each response."
            prompt += f"UserQ: {question}"
            prompt += str(mygraph.get_triples()).strip('[]')
            response = ollama_stream(prompt, selected_model)
            full_response = ''
            for token in response:
                full_response += token
            st.write(full_response)

        #st.markdown(question)

        #st.rerun(scope="fragment")

if st.button("Find!"):
    #full_response = "### here are  some datasets that I found, related to your question:   \n"

    subset, edge_index, _, edge_mask = k_hop_subgraph(qa_dict[selected_q], num_hops=1, edge_index=st.session_state.skb.edge_index)

    g_node_types = st.session_state.skb.node_types[subset]
    g_edge_types = st.session_state.skb.edge_types[edge_mask]

    G = nx.DiGraph()

    node_list = subset.numpy()
    edge_list = edge_index.transpose(1, 0).numpy()
    edge_types = g_edge_types.numpy()

    for n in node_list:
        G.add_node(st.session_state.skb[n].name)
    for i, e in enumerate(edge_list):
        G.add_edge(st.session_state.skb[e[0]].name, st.session_state.skb[e[1]].name, relation=st.session_state.skb.edge_type_dict[edge_types[i]], label=st.session_state.skb.edge_type_dict[edge_types[i]])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    fig = gv.d3(G,

        show_edge_label=True,

        edge_label_data_source='label',

        edge_curvature=0.25,

        zoom_factor=2.0,

        many_body_force_strength=-1000)

    components.html(fig.to_html(), height=600)
    do_qa()
    
        