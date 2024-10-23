"""
Streamlit Application for Running and Analyzing BioModel Simulations

This code provides an interactive web-based interface for users to run and analyze 
biological simulations of glycolysis models using Streamlit. The application allows 
users to:

1. Select a biological model (specifically from the Teusink Glycolysis models).
2. Input species and timeline details for the simulation.
3. Generate and visualize simulation results including time-course plots, 
   species' initial concentrations, and model metadata.
4. Ask questions about the generated simulation results using an integrated 
   large language model (LLM) via OpenAI's API and LangChain.

The following libraries and functionalities are used:
- **basico**: For running simulations of biological models.
- **Streamlit**: For creating an interactive user interface.
- **Pandas**: For handling tabular simulation data.
- **PIL**: For handling images (simulation plots).
- **LangChain and OpenAI API**: For generating responses to user questions based on the simulation data.

Users can select a model ID, provide species information and timeline, run simulations, 
and then visualize and explore the results interactively. The results are saved to CSV 
files, and the application provides a summary of the simulation along with a graphical plot.

Example prompt: Run the model with ID 64 for 100 minutes. Please make sure the initial concentration of "Glucose in Cytosol" is 10.0.
Example prompt: What species has the highest concentration?
"""

import streamlit as st
# Session state to keep track of the visibility state
if "show_text" not in st.session_state:
    st.session_state.show_text = False

import os
os.environ["OPENAI_API_KEY"] = "XXX"

import pandas as pd
import os
import glob
from PIL import Image
from trigger_basico import generate_simulation
from extract_id import get_model_id, get_duration_time, get_species_info, get_ids, remove_all_files
st.set_page_config(page_title="LLM-RAG-for-biology-team-5", page_icon="🌟", layout="wide")

st.title("Ask me about a biomedical computational model")
st.write(
        "Talk to BioModels: The LLM will run a computational model based on your input."
    )
# selected_model = st.selectbox("Select a model:", models, index=models.index(default_model))


col1, col2 = st.columns([1, 2])

with col1:

    # model_inp_text = st.selectbox("Model ID ?",(64, 297, 535))
    user_inp_text = st.text_area("Specify the BioModels ID and parameters you want to simulate:", height=100)
    ids_json = get_ids(user_inp_text)
    model_inp_text = ids_json["id"]
    try:
        species_name= ids_json["species"]
        species_concentraion= float(ids_json["concentration"])
    except:
        species_name = None
        species_concentraion = None

    duration_inp_text = ids_json["duration"]
    

    if st.button("generate simulation", type="primary"):
        st.write(f"Selected BIO model : {model_inp_text}")
        st.write(f"Selected duration : {duration_inp_text}")
        st.write(f"Selected species : {species_name}")
        st.write(f"Selected concentraion : {species_concentraion}")
        generate_simulation(model_id=model_inp_text, species_name=species_name, species_concentraion =species_concentraion, duration=int(duration_inp_text))
    
    st.divider()


try:
    from pathlib import Path
    result_path = Path(__file__).resolve().parent.parent.parent / "results"
    df = pd.read_csv(f'{result_path}/glycolysis_simulation.csv')
    
    input_species_df = pd.read_csv(f'{result_path}/input_species.csv')[["sbml_id", "initial_concentration"]]

    image = Image.open(f'{result_path}/glycolysis_simulation.png')

    with open(f"{result_path}/name.txt", "r") as file:
        model_name = file.read()


    with open(f"{result_path}/description.html", "r", encoding="utf-8") as file:
        html_content = file.read()

except:
    image = None
    pass

from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI



with col2:
    # Display the image
    if image:
        question = st.text_input("Ask a question about the generated simulation results:")
        csv_path = str(Path(__file__).resolve().parent.parent.parent / "results/glycolysis_simulation.csv")
        if st.button("ask question", type="primary"):
            st.cache_resource.clear()
            agent = create_csv_agent(OpenAI(temperature=0), allow_dangerous_code=True, path=csv_path)
            if question and image:
                llm_result = agent.invoke(question)
                st.write(llm_result["output"])
            else:
                st.write("Please Generate a simulation or Enter a Question")

        if st.button("clear", type="primary"):
            st.cache_resource.clear()
            remove_all_files(result_path)
            st.write("Cleared the cache")


        st.header(f"{model_name}")

                # Toggle visibility when the button is clicked
        if st.button("Show/Hide Text"):
            st.session_state.show_text = not st.session_state.show_text

        # Conditionally display the text
        if st.session_state.show_text:
            st.image(image, caption='Displayed Image')

            st.header("summary : ")
            st.html(html_content)

            st.header("List of Simulated Speices : ")
            st.table(input_species_df)

        
