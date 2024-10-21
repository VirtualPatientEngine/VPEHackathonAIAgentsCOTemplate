import streamlit as st
import pandas as pd
import ollama
from ollama_utils import ollama_stream, ollama_response
from query_utils import from_collection_datasets_to_markdown, extract_dictionary_from_response
from gene_expression_datasets_rag import load_rag_system, search_by_question
from AD_QSP_tools import test_drug_efficacy, get_easi_severity, question_examples
from cell2Sentence import question_to_cell_type

st.set_page_config(page_title="LLM-RAG-for-biology-examples", page_icon="🌟", layout="wide")
load_rag_system()


# ----------------------------------------------------------
# --- Section: Streamlit App ---

default_model = "llama3.1:latest"
models = [m["name"] for m in ollama.list()["models"] if not "embed" in m["name"]]
st.title("Ask me bio-medical question")
selected_model = st.selectbox("Select a model:", models, index=models.index(default_model))


# Create tabs - one for each model: RAG and sysBio
rag_tab, ecsema_tab, cell2sentence_tab = st.tabs(
    ["Ask about Single-Cell RNA-seq data", "Talk with SysBio Model", "Ask about cells"]
)

# --------------------------------------------------------
# --- Section: RAG Model ---
with rag_tab:
    # Button to generate response
    st.write(
        "RAG model: The LLM will search for datasets related to the question, and explain them in a professional way."
    )
    question = st.text_input("Describe the single cell expression question:")
    if st.button("Find a datasets!"):
        with st.spinner("Searching for datasets..."):
            placeholder = st.empty()
            full_response = "### here are  some datasets that I found, related to your question:   \n"

            ids, docs = search_by_question(question)
            for id_, document in zip(ids, docs):

                # get the collection from the table where the index row is the id_:
                data = st.session_state.dataset.iloc[int(id_)]
                collection_name = data["name"]  # get the collection name
                collection_id = data["id"]  # get the collection id
                datasets = pd.DataFrame(data["datasets"].tolist())  # get the datasets from the collection
                table = from_collection_datasets_to_markdown(datasets)

                full_response += f"\n#### Gene expression collection:[{collection_name}](https://cellxgene.cziscience.com/collections/{collection_id})  \n"
                full_response += f"\n{table}   \n\n\n----   \n"
                prompt = f"The user ask: '{question}', Summarize the the documents in a professional and clear way with one paragraph, in a way that is related to the users question. :\n{document}\n start your response with: This datasets is about"

                response = ollama_stream(prompt, selected_model)
                for token in response:
                    full_response += token
                    placeholder.markdown(full_response)

# --------------------------------------------------------
# --- Section: SysBio Model ---
with ecsema_tab:
    st.markdown(
        """This tab is for the **SysBio Model**: *Miyano et al. - A Mathematical Model for Drug Targets in Dupilumab Poor Responders (AD-QSP Model)*.

This model helps identify the best drug combinations for patients with **eczema** who don’t respond well to **dupilumab**. By simulating how different drugs affect the body, it predicts which ones can reduce eczema severity.

**Want to explore a drug? Just ask!** I’ll assess how it may impact eczema severity by adjusting key factors:

- **Skin Barrier Integrity**: Protects against damage/infection.
- **Infiltrated Pathogens**: Level of harmful microorganisms.
- **Th1, Th2, Th17, Th22**: Immune cells causing inflammation.
- **IL-4, IL-13, IL-17, IL-22, IL-31**: Inflammatory cytokines, drug targets.
- **IFNg, TSLP, OX40L**: Additional immune and inflammation factors.

I'll simulate the effects of a drug on these parameters, starting from `0` (no effect), and predict how it influences eczema severity.
"""
    )

    question = st.text_input("Ask me a question about the drugs:")

    # Add a blank option to the selectbox and use it to reset behavior
    question_selected = st.selectbox("Select a question example:", [""] + question_examples, index=0)

    # If the user selects a question from the selectbox, override the text input with the selected question
    if question_selected != "":
        question = question_selected

    if st.button("Ask me a question!"):
        with st.spinner("⚙️ Extracting parameters..."):
            placeholder = st.empty()
            full_response = ""
            prompt = f"""
You are tasked with generating a Python dict that represents the effects of a drug on specific biological parameters within the AD-QSP model. 
Follow these steps precisely:

<instructions>
1. Always return the Python dictionary enclosed in ``` signs.
2. The dictionary must only include the parameters: "EASI score", "Skin barrier integrity", "Infiltrated pathogens", "Th1", "Th2", "Th17", "Th22", "IL-4", "IL-13", "IL-17", "IL-22", "IL-31", "IFNg", "TSLP", "OX40L"
   - **Do not include** any parameter that is not mentioned.
3. For each mentioned parameter, infer the value based on user input:
   - If a percentage change is given, apply it as a decimal (e.g., 30% reduction → `-0.3`).
   - For descriptive terms:
     - "slight" or "small": ±0.1
     - "moderate": ±0.3
     - "significant" or "large": ±0.5
4. If the user is unsure or doesn't specify the effect size, assign `0`.
5. Return only the mentioned parameters in the final dictionary.
6. IMPORTANT: do not generate any python code just a strict dictionary !
</instructions>
<examples>
input:
"This drug lowers IL-4 by 50% and slightly increases Th1."
response:
```
{{
    "IL-4": -0.5,
    "Th1": 0.1
}}
```

input:
"It increases Th2 and reduces IL-22 slightly."

response:
```
{{
    "Th2": 0.3,
    "IL-22": -0.1
}}
```

input:
"The drug greatly reduces IL-17 and has no effect on other factors."

response:
```
{{
    "IL-17": -0.5
}}
```
</examples>
Be strict and precise in following these instructions.
input:
{question}
response:
```
"""
            result = ollama_response(prompt, selected_model)
            params = extract_dictionary_from_response(result)

        with st.spinner("🚀 Running simulation..."):  # a good 5 different emojis for running simulation: 🏃‍♂️,
            sim_result = test_drug_efficacy(params)
            mean_easi = sim_result["mean_easi"][-1]
            std_easi = sim_result["std_easi"][-1]
            severity = get_easi_severity(mean_easi)
            full_response += (
                f"### Drug effects:  EASI (24 weeks) : {mean_easi:.2f} ± {std_easi:.2f} (mean ± std)  \n        "
            )
            full_response += f"\nExpected Severity:  **{severity}**  \n      "
            full_response += "The Mean EASI is the average score that shows how severe eczema symptoms are for a group of people, with higher numbers indicating worse symptoms and lower numbers showing improvement. \n      "

            st.write(full_response)

# --------------------------------------------------------
# --- Section: Cell2Sentence Model ---
with cell2sentence_tab:
    st.write(
        'This code allows users to input questions like "What cell types are associated with genes MT-CO3 and MT-ATP6?" and automatically generates a clear, professional response identifying the relevant cell types or expected genes based on the provided information.'
    )
    question = st.text_input("Ask me a question about cells or a list of genes:")
    if st.button("Ask about cells"):
        with st.spinner("🔍 Analyzing the question..."):
            placeholder = st.empty()
            full_response = question_to_cell_type(question)
            st.write(full_response)
