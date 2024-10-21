import numpy as np
from scipy.integrate import solve_ivp
from libsbml import SBMLReader
from pathlib import Path
import matplotlib.pyplot as plt

# Function to load SBML model
def load_model(file_path):
    reader = SBMLReader()
    document = reader.readSBML(file_path)
    
    if document.getNumErrors() > 0:
        raise Exception(f"SBML Errors: {document.getErrorLog().toString()}")
    
    model = document.getModel()
    if model is None:
        raise ValueError("No model could be retrieved from the SBML file.")
    
    return model

# Function to extract species and their initial concentrations (with non-zero initial conditions)
def extract_species_and_initial_conditions(model):
    species_list = []
    initial_conditions = []
    
    for i in range(model.getNumSpecies()):
        species = model.getSpecies(i)
        species_list.append(species.getId())
        
        # Randomize the initial conditions for some species
        if i % 2 == 0:
            initial_conditions.append(np.random.uniform(0.1, 1.0))  # Random initial concentration for even-indexed species
        else:
            initial_conditions.append(0.0)  # Keep some species at 0 for diversity
        
        print(f"Initialized {species.getId()} with concentration {initial_conditions[-1]}")
    
    return species_list, initial_conditions

# Function to define dynamic reaction rates with randomness and interactions
def reaction_rates(t, y, species_list):
    dydt = []
    
    for i, conc in enumerate(y):
        # Random factor to simulate environmental noise (-0.05 to 0.05)
        random_factor = np.random.uniform(-0.05, 0.05)
        
        if i % 2 == 0:
            # Exponential decay for some species
            rate = (-0.1 + random_factor) * conc
        else:
            # Exponential growth for other species, influenced by another species
            related_species = y[i-1] if i > 0 else 0.1  # Growth depends on the previous species' concentration
            rate = (0.2 + random_factor) * conc + 0.1 * related_species  # Growth with interaction from another species
        
        dydt.append(rate)
        
        # Print current state at specific time intervals
        if int(t) % 20 == 0 and t > 0:
            print(f"At time {t:.2f}: {species_list[i]} concentration is {conc:.4f}")
    
    return dydt

# Function to run the simulation
def run_simulation(model, species_list, initial_conditions):
    t_span = (0, 100)  # Time span from 0 to 100
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # Time points
    
    # Solve the ODE with reaction rates
    solution = solve_ivp(reaction_rates, t_span, initial_conditions, t_eval=t_eval, args=(species_list,))
    
    return solution

# Function to plot simulation results
def plot_results(solution, species_list, model_name):
    plt.figure(figsize=(10, 6))
    
    for i, species in enumerate(species_list):
        plt.plot(solution.t, solution.y[i], label=species)
    
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"Dynamic Simulation of {model_name} with Interactions and Randomness")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Main function to orchestrate the entire simulation
def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    file_path = f"{data_dir}/BIOMD0000000015_url.xml"
    
    # Load the SBML model
    model = load_model(file_path)
    print(f"Model loaded: {model.getName()}")
    
    # Extract species and initial conditions
    species_list, initial_conditions = extract_species_and_initial_conditions(model)
    
    # Run the simulation
    solution = run_simulation(model, species_list, initial_conditions)
    
    # Plot the results
    plot_results(solution, species_list, model.getName())

if __name__ == "__main__":
    main()