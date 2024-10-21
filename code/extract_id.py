import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
import glob

os.environ["OPENAI_API_KEY"] = "XXX"

def get_model_id(text = "Hi,  Make  simulation of ID with 64 ."):
    """
    Extract the model ID number from the given text and return it as a JSON object.
    """

    # Initialize the OpenAI chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo")  # or "gpt-4"

    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_template("""
    Extract the Model ID number from the following text and return it in JSON format:
    "{text}"
    For example,
    Questions is : Please use Model ID 64 and generate simulation.
    Annswer should be : {{"id" : 64}}              
    """)


    # Create a chain for chat-based models
    chain = LLMChain(llm=chat, prompt=prompt)


    # Run the chain
    response = chain.run(text)

    # Parse the response as JSON
    response_json = json.loads(response)

    return response_json["id"]

def get_species_info(text="Keep the concentration of Sic to 0.005."):
    """
    Extract the species name and concentration from the given text and return it as a JSON object.
    """

    # Initialize the OpenAI chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo")  # or "gpt-4"

    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_template("""
    Extract the species name and concentration from the following text and return it in JSON format:
    "{text}"
    
    For example:
    Question: "."
    Answer: {{"species": "Sic", "concentration": 0.005}}
    """)

    # Create a chain for chat-based models
    chain = LLMChain(llm=chat, prompt=prompt)

    # Run the chain
    response = chain.run(text)
    print("Raw response from model:", response)

    try:
        # Parse the response as JSON
        response_json = json.loads(response)
        print("Parsed JSON:", response_json)
        return response_json["Species"]
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def get_duration_time(text = "Run for 100 minutes."):

    """
    Extract the duration time from the given text and return it as a JSON object.
    """
    # Initialize the OpenAI chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo")  # or "gpt-4"

    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_template("""
    Extract the duration time from the following text and return it in JSON format:
    "{text}"
    For example,
    Questions is : Run the simulaiton for 100 minutes.
    Annswer should be : {{"duration" : 100}}              
    """)


    # Create a chain for chat-based models
    chain = LLMChain(llm=chat, prompt=prompt)


    # Run the chain
    response = chain.run(text)
    print(response)
    # Parse the response as JSON
    response_json = json.loads(response)
    
    return response_json["duration"]


def get_ids(text = "Run the model with ID 64 for 100 minutes. Please make sure the initial concentration of Sic is 0.005."):
    """
    Extract the model ID, duration time and species initial concentration from the given text and return it as a JSON object.
    """
    # Initialize the OpenAI chat model
    chat = ChatOpenAI(model="gpt-3.5-turbo")  # or "gpt-4"

    # Define a chat prompt template
    prompt = ChatPromptTemplate.from_template("""
    Extract the model ID, duration time and species initial concentration from the following text and return it in JSON format:
    "{text}"
    For example,
    Questions is : Run the model with ID 64 for 100 minutes. Please make sure the initial concentration of Sic is 0.005.
    Annswer should be : {{"id" : 64, "duration" : 100, "species": "Sic", "concentration": 0.005}}
    """)


    # Create a chain for chat-based models
    chain = LLMChain(llm=chat, prompt=prompt)


    # Run the chain
    response = chain.run(text)
    # Parse the response as JSON
    response_json = json.loads(response)
    
    return response_json
# get_duration_time()
# get_model_id()
# get_species_info()
# get_ids()

def remove_all_files(directory):
    """
    Remove all files in the specified directory.
    """
    # Get all file paths in the directory
    files = glob.glob(os.path.join(directory, '*'))

    # Iterate over the list and remove each file
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")