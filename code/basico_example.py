"""
This study explores whether yeast glycolysis can be understood using the 
in vitro kinetics of its enzymes. The model replicates the steady-state 
fluxes and metabolite concentrations from Teusink et al. (2000). It 
includes ATP consumption with the same stoichiometry and has been 
validated using Copasi. Vmax values were converted with a factor of 270, 
and the ADH equilibrium constant was adjusted for the forward reaction. 
The results closely align with the original study, with slight differences 
in G6P concentration. Some corrections were made to kinetic equations. 
ATP, ADP, and AMP species were added for easier curation. The model is 
in the public domain (CC0), allowing free use and modification. Cite the 
BioModels Database via Li et al. (2010).

In the code, the `basico` library loads the Teusink2000 Glycolysis model 
([BIOMD0000000064](https://www.omicsdi.org/dataset/biomodels/BIOMD0000000064)). 
It retrieves species' concentrations and parameters, removes amount 
expressions, and runs a time-course simulation using SBML IDs. This setup 
enables dynamic exploration of the model for better understanding of 
glycolysis and enzyme kinetics.
"""

from pathlib import Path
from basico import load_biomodel, get_species, get_parameters, run_time_course, remove_amount_expressions
import matplotlib.pyplot as plt
result_path = Path(__file__).resolve().parent.parent / "results"

load_biomodel(64)  # Load the BIOMD0000000064 (Teusink2000_Glycolysis) model
species = get_species() # Get the species and their initial concentrations
params = get_parameters() # Get the parameters and their values
remove_amount_expressions()  # Remove amount expressions to use the species concentrations directly
result = run_time_course(use_sbml_id=True)  # Run the simulation with the SBML model for 100 time units
result.to_csv(f"{result_path}/glycolysis_simulation.csv")

# Plot the simulation results over time:
axes = result.plot(figsize=(12, 6), title="Teusink2000 Glycolysis Model Simulation")
plt.savefig(f"{result_path}/glycolysis_simulation.png")

