from pathlib import Path
import basico
from basico import load_biomodel, get_species, set_species, get_parameters, run_time_course, remove_amount_expressions
from basico.biomodels import get_model_info
import matplotlib.pyplot as plt
result_path = Path(__file__).resolve().parent.parent.parent / "results"

def generate_simulation(model_id=297, species_name=None, species_concentraion=None, duration=5):

    """
    Generates a time-course simulation for the specified BioModel.

    This function loads the Teusink2000 Glycolysis model (BIOMD0000000064) 
    from the BioModels Database, retrieves species concentrations and model 
    parameters, removes amount expressions, and runs a time-course simulation. 
    The results are saved to CSV and PNG files.

    Parameters:
    -----------
    model_id : int, optional
        The BioModels Database ID for the model to simulate. 
        Default is 297, corresponding to BIOMD0000000064 (Teusink2000_Glycolysis).

    Outputs:
    --------
    1. Writes the model name to `name.txt`.
    2. Writes the model description to `description.html`.
    3. Saves species' initial concentrations to `input_species.csv`.
    4. Runs a time-course simulation and saves results to `glycolysis_simulation.csv`.
    5. Plots the simulation results and saves the plot to `glycolysis_simulation.png`.
    
    Example:
    --------
    To run the simulation with the default Teusink2000 Glycolysis model:
    >>> generate_simulation(model_id=297)
    """
    
    model_obj = load_biomodel(model_id)  # Load the BIOMD0000000064 (Teusink2000_Glycolysis) model

    #print(basico.get_model_name(model=model_obj))
    with open(f"{result_path}/name.txt", "w") as file:
        file.write(get_model_info(model_id)["name"])


    #species = get_species() # Get the species and their initial concentrations
    #print(species.head())
    #print(species_name, species_concentraion)
    if species_name != None and species_concentraion != None:
        set_species(name=species_name, initial_concentration=species_concentraion, model=model_obj)


    with open(f"{result_path}/description.html", "w", encoding="utf-8") as file:
            file.write(get_model_info(model_id)["description"])
    


    ## Setting the patient informaiton
    # https://basico.readthedocs.io/en/latest/notebooks/Setting_up_Parameter_Estimation.html
    params = get_parameters() # Get the parameters and their values
    get_species().to_csv(f"{result_path}/input_species.csv")
    remove_amount_expressions()  # Remove amount expressions to use the species concentrations directly
    result = run_time_course(use_sbml_id=True, start_time=0, interval=0.1, duration=duration)

    result.to_csv(f"{result_path}/glycolysis_simulation.csv")

    # Plot the simulation results over time:
    axes = result.plot(figsize=(12, 6), title="Model Simulation")
    plt.savefig(f"{result_path}/glycolysis_simulation.png")

# generate_simulation(model_id=297)