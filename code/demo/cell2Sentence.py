
#cell2Sentence.py
"""
This Python script processes user inquiries related to cell types and associated genes by integrating a pre-trained causal language model with the Ollama API. 
It intelligently determines whether a user's query specifies cell types or gene lists, then generates relevant prompts to extract expected genes or predict cell types accordingly. 
Utilizing the Hugging Face Transformers library, the script tokenizes and generates text based on the input, while the Ollama API refines these outputs into concise, 
professional responses. Designed for applications in biological data analysis, 
It facilitates accurate interpretation and communication of complex gene-cell type relationships by automating the response generation process.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from ollama_utils import ollama_response

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

cache_path = Path(__file__).resolve().parents[1] / "scratch/transformers"
# c2s_model = "vandijklab/pythia-160m-c2s"
c2s_model = "vandijklab/C2S-Pythia-410m-cell-type-prediction"

tokenizer = AutoTokenizer.from_pretrained(c2s_model, cache_dir=cache_path)
model = AutoModelForCausalLM.from_pretrained(c2s_model, cache_dir=cache_path).to(device)


def cell2Sentence(query: str, max_length=100) -> str:
    inputs = tokenizer(query, return_tensors="pt").to(device)
    tokens = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(tokens[0])


def question_to_cell_type(query: str) -> str:
    intitial_prompt = f"""
<user-query> {query} </user-query>
<task>
Identify if the user specified a cell type or a list of genes.
If the user provided a cell type, return a specific prompt wrapped in a <prompt> tag as follows:
<prompt>The expected genes based on these cell types are {{cell_types}}</prompt>
If the user provided a list of genes, return the following prompt:
<prompt>{{genes_list_first}} The expected cell type based on these genes is</prompt>
! IMPORTANT !: Only return the <prompt> tag. Do not add any additional information, comments, or code.
</task>
"""
    print("Sending the prompt to the Ollama API")
    result = ollama_response(intitial_prompt)
    # extract the prompt from the response:
    result = result.split("<prompt>")[1].split("</prompt>")[0]
    print("Using the cell2Sentence model to generate the response")
    result = cell2Sentence(result)
    print("Raw result: ", result)
    final_prompt = f"""
<user-query> {query} </user-query>
<model-answer> {result} </model-answer>
<task>
based on the model answer construct a short and professional response to the user query
</task>    
"""
    result = ollama_response(final_prompt)
    return result


if __name__ == "__main__":
    # result = cell2Sentence(
    #     "MT-CO3 MT-ATP6 MT-CYB MT-ND4L MT-ND3 MT-ND1 MT-ND4 MT-ND5 MT-ATP8 NEAT1 MT-ND2 FOS S100A8 S100A9. The expected cell type based on these genes is"
    # )

    result = question_to_cell_type("what will be the cell types where thr top genes are MT-CO3 MT-ATP6 MT-CYB MT-ND4L MT-ND3 MT-ND1?")
    print(result)

