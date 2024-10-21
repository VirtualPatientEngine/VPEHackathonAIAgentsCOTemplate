import streamlit as st
from Antibody_dose import set_dose_and_run_simulation

# Using the title function to set a title for your app
st.title("Dose simulation and analysis")

# Getting the user dose using text_input and converting it to float
user_dose = st.text_input("Please enter the dose (default is 300): ", '300')
user_dose = float(user_dose)

# Button to run simulation
if st.button('Run Simulation and Generate Plot'):
    try:
        # Running the function and capturing the plot output and crp_fold_change
        plot, crp_fold_change = set_dose_and_run_simulation(user_dose)
        
        # Displaying the CRP fold change in Streamlit
        st.write('CRP log2 fold change:', crp_fold_change)
        
        # Displaying the matplotlib plot in Streamlit
        st.pyplot(plot.figure)
    except Exception as e:
        st.error(f'An error occurred during the simulation: {str(e)}')
