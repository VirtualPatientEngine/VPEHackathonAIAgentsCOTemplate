# BioModel Simulation and Analysis Web App

This Streamlit-based web application allows users to run and analyze biological simulations, particularly focused on glycolysis models. The app provides an interactive interface to select models, input simulation parameters, generate results, and visualize the output. It also integrates a language model for querying and explaining the simulation outcomes.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

## Features

- **Model Selection:** Choose a biological model, specifically from the Teusink Glycolysis models, using a model ID.
- **Simulation Setup:** Input parameters such as species information and duration for the simulation.
- **Run Simulations:** Execute the simulations using the `basico` library for modeling.
- **Visualization:** View results through time-course plots, species' initial concentrations, and model metadata.
- **Query Results:** Ask questions about the simulation results using a language model (LLM) integrated via OpenAI's API and LangChain.
- **Export & Clear Results:** Save results as CSV files and provide options to clear cache and data.

## Technologies Used

The application leverages several libraries and tools to provide a seamless experience:
- **[Streamlit](https://streamlit.io/):** For creating an interactive web interface.
- **[basico](https://pypi.org/project/basico/):** To run biological model simulations.
- **[Pandas](https://pandas.pydata.org/):** For handling data manipulation and tabular results.
- **[Pillow (PIL)](https://pillow.readthedocs.io/):** For image processing and visualization.
- **[LangChain](https://langchain.readthedocs.io/):** For building language model-driven functionalities.
- **[OpenAI API](https://platform.openai.com/docs/):** To integrate a large language model for natural language processing and question-answering capabilities.

## Prerequisites

Before running the application, ensure you have the following:
1. **Python 3.8 or higher**
2. **Required libraries** (install using the following command):
   ```bash
   pip install streamlit pandas pillow basico langchain openai

 #OpenAI API Key: Set up an environment variable for the API key:

  export OPENAI_API_KEY='your-openai-api-key-here'
  
## Installation
Clone the repository:
git clone https://github.com/your-repository-url/bio-simulation-app.git

Navigate to the project directory:
cd bio-simulation-app

## Install dependencies:

bash
Copy code
pip install -r requirements.txt

## Usage

Run the Streamlit application:
streamlit run app.py
Select a biological model:

Provide a Model ID in the input field (e.g., 64, 297, 535).
Input simulation parameters:

Specify the species name, concentration, and simulation duration.
Generate simulation:

Click the "Generate Simulation" button to run the simulation.
View and analyze the results:

If a simulation was generated successfully, the results are displayed, including graphical plots and a summary of the simulated species.
Ask questions about the simulation:

Enter a question related to the generated results, and the integrated language model will provide an explanation.
Clear cache and results:

Use the "Clear" button to remove all cached data and start fresh.

## Folder Structure

bio-simulation-app/
├── app.py                  # Main application file
├── requirements.txt        # List of required libraries
├── results/                # Directory for storing simulation results
│   ├── glycolysis_simulation.csv
│   ├── input_species.csv
│   ├── glycolysis_simulation.png
│   ├── name.txt
│   └── description.html
├── trigger_basico.py       # Script for running simulations
└── extract_id.py           # Script for extracting model and species info

## Configuration

Setting up OpenAI API Key:
The API key for OpenAI is required to enable the language model functionality. Make sure to set it as an environment variable.
Streamlit Settings:
Customize the Streamlit settings in app.py (e.g., page title, layout).

## Troubleshooting

Simulation Not Generating:
Verify that the model ID and species information are correct.
Ensure that the basico library is installed and properly configured.
OpenAI API Issues:
Make sure the API key is correctly set up in the environment.
Check for any API usage limits.

## Future Enhancements

Add More Models: Expand the application to support additional biological models.
Advanced Analysis: Incorporate more advanced data analysis and visualization techniques.
Enhanced Error Handling: Improve error handling to provide more descriptive messages.


## Acknowledgments

Streamlit
OpenAI
LangChain
