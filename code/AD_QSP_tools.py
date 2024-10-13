"""
**AD_QSP_tools.py**

This module implements a mathematical model for simulating drug effects on eczema severity using the Atopic Dermatitis Quantitative Systems Pharmacology (AD-QSP) model, based on the work of Miyano et al. 
Its primary purpose is to provide a computational tool that predicts how different drugs influence the progression of atopic dermatitis by affecting various biological factors. 
The model quantifies eczema severity using the Eczema Area and Severity Index (EASI) score, enabling researchers and clinicians to assess potential drug efficacy in a controlled, simulated environment.

The implementation centers around a system of ordinary differential equations (ODEs) that represent the complex interactions between key biological components involved in eczema, 
such as skin barrier integrity, infiltrated pathogens, immune cells (Th1, Th2, Th17, Th22), and cytokines (e.g., IL-4, IL-13, IL-17). 
The `diff_eq` function defines these equations, incorporating drug effects by adjusting the concentrations of specific factors based on user-defined inputs. 
An `ODE` class is provided to numerically solve these equations over time using SciPy's `odeint` function, allowing for dynamic simulations of disease progression under various drug influences.

To use the module, users define drug effects through a dictionary where keys are biological factors and values 
represent the fractional change induced by the drug (e.g., `{"IL-4": -0.5}` for a 50% reduction in IL-4 levels). 
The `test_drug_efficacy` function simulates these effects across a cohort of virtual patients (defaulting to 1,000), 
each characterized by parameter sets sampled from statistical distributions to reflect biological variability. 
This function returns the mean and standard deviation of EASI scores over the simulation period, facilitating the evaluation of average drug efficacy and response variability within the population. 
Additionally, the `get_easi_severity` function interprets EASI scores to classify eczema severity levels, providing a qualitative assessment alongside quantitative results.
"""


import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Default drug effects dictionary
default_drug_effects = {
    "EASI score": 0,  # No direct effect on EASI score
    "Skin barrier integrity": 0,
    "Infiltrated pathogens": 0,
    "Th1": 0,
    "Th2": 0,
    "Th17": 0,
    "Th22": 0,
    "IL-4": 0,
    "IL-13": 0,
    "IL-17": 0,
    "IL-22": 0,
    "IL-31": 0,
    "IFNg": 0,
    "TSLP": 0,
    "OX40L": 0,
}


def load_parameters(mu_file="/code/AD_QSP_model/mu.csv", sigma_file="/code/AD_QSP_model/sigma.csv") -> tuple:
    """Load the mu and sigma parameters from CSV files."""
    mu = pd.read_csv(mu_file, header=None).to_numpy().flatten().reshape(-1, 1)
    sigma = pd.read_csv(sigma_file, header=None).to_numpy().flatten().reshape(-1, 1)
    return mu[:51], sigma[:51]


def EASI(sim):
    s = sim[:, 0]
    p = sim[:, 1]
    e = 72 * (2 * p + 2 * (1 - s)) / 4
    return e


def diff_eq(c, t, de, x):
    # Prepare parameter values
    k1 = x[0]
    k2 = x[1]
    k3 = min([x[2], de[0]])
    b1 = x[3]
    b2 = x[4]
    b3 = x[5]
    b4 = x[6]
    b5 = x[7]
    d1 = x[8]
    d2 = x[9]
    d3 = x[10]
    b6 = x[11]
    d4 = x[12]
    d5 = x[13]
    d6 = x[14]
    d7 = x[15]
    b7 = x[16]
    b8 = x[17]
    d8 = x[18]
    k5 = x[19]
    k9 = x[20]
    d9 = x[21]
    b9 = x[22]
    k6 = x[23]
    k10 = x[24]
    k7 = x[25]
    k8 = x[26]
    k11 = x[27]
    k12 = x[28]
    d10 = x[29]
    k13 = x[30]
    k14 = x[31]
    d11 = x[32]
    k15 = x[33]
    k16 = x[34]
    d12 = x[35]
    k17 = x[36]
    k18 = x[37]
    d13 = x[38]
    k19 = x[39]
    k20 = x[40]
    d14 = x[41]
    k21 = x[42]
    k22 = x[43]
    d15 = x[44]
    k23 = x[45]
    k24 = x[46]
    d16 = x[47]
    k25 = x[48]
    k26 = x[49]
    d17 = x[50]
    ea2 = max([0.4396, de[9]])
    k4 = d8

    # Effective concentration of cytokines (drug effects on cytokines)
    IL4 = (1 - de[1]) * c[6]
    IL13 = (1 - de[2] * ea2) * c[7]
    IL17 = (1 - de[3]) * c[8]
    IL22 = (1 - de[4]) * c[9]
    IL31 = (1 - de[5]) * c[10]
    TSLP = (1 - de[6]) * c[12]
    OX40 = (1 - de[7]) * c[13]
    IFNg = c[11] + de[8]

    # ODEs
    dc0dt = (1 - c[0]) * (k1 + k2 * IL22 + k3) / (
        (1 + b1 * IL4) * (1 + b2 * IL13) * (1 + b3 * IL17) * (1 + b4 * IL22) * (1 + b5 * IL31)
    ) - c[0] * (d1 * (1 + d3 * c[1]) + d2 * IL31)
    dc1dt = k4 / (1 + b6 * c[0]) - c[1] * (
        ((1 + d4 * c[1]) * (1 + d5 * IL17) * (1 + d6 * IL22) * (1 + d7 * IFNg)) / ((1 + b7 * IL4) * (1 + b8 * IL13))
        + d8
    )
    dc2dt = k5 * c[1] * (1 + k9 * IFNg) / (4 + k9 * IFNg + k10 * IL4) - d9 * c[2] / (1 + b9 * OX40)  # Th1
    dc3dt = k6 * c[1] * (1 + k10 * IL4) / (4 + k9 * IFNg + k10 * IL4) - d9 * c[3] / (1 + b9 * OX40)  # Th2
    dc4dt = k7 * c[1] / (4 + k9 * IFNg + k10 * IL4) - d9 * c[4] / (1 + b9 * OX40)  # Th17
    dc5dt = k8 * c[1] / (4 + k9 * IFNg + k10 * IL4) - d9 * c[5] / (1 + b9 * OX40)  # Th22
    dc6dt = k11 * c[3] + k12 - d10 * c[6]  # IL4
    dc7dt = k13 * c[3] + k14 - d11 * c[7]  # IL13
    dc8dt = k15 * c[4] + k16 - d12 * c[8]  # IL17
    dc9dt = k17 * c[5] + k18 - d13 * c[9]  # IL22
    dc10dt = k19 * c[3] + k20 - d14 * c[10]  # IL31
    dc11dt = k21 * c[2] + k22 - d15 * c[11]  # IFNg
    dc12dt = k23 * c[1] + k24 - d16 * c[12]  # TSLP
    dc13dt = k25 * TSLP + k26 - d17 * c[13]  # OX40L

    dcdt = np.array(
        [dc0dt, dc1dt, dc2dt, dc3dt, dc4dt, dc5dt, dc6dt, dc7dt, dc8dt, dc9dt, dc10dt, dc11dt, dc12dt, dc13dt]
    )
    return dcdt


class ODE(object):
    def __init__(self, diff_eq, init_con):
        self.diff_eq = diff_eq
        self.init_con = init_con

    def cal_equation(self, t_end, drug_effect, x):
        dt = 0.1  # delta time (week)
        N = round(t_end / dt) + 1  # Time steps
        t = np.linspace(0, t_end, N)  # prepare time
        v = odeint(self.diff_eq, self.init_con, t, rtol=1e-8, atol=1e-6, args=(drug_effect, x))
        return v


def simulate_patient(x, de):
    # Initial conditions for simulating steady-state levels of biological factors
    init_cond = np.array(
        [
            0.5931,  # s_0: Skin barrier integrity
            0.4069,  # p_0: Infiltrated pathogens
            3.1,  # Th1_0
            8.7,  # Th2_0
            2.0,  # Th17_0
            21.0,  # Th22_0
            38.0,  # IL4_0
            40.5,  # IL13_0
            5.4,  # IL17_0
            3.0,  # IL22_0
            2.0,  # IL31_0
            1.5,  # IFNg_0
            4.4,  # TSLP_0
            9.7,  # OX40_0
        ],
        dtype="float64",
    )

    # Simulate steady state
    ode = ODE(diff_eq, init_cond)
    drug_effect_ss = np.zeros(10, dtype="float64")  # No drug effects for steady state
    drug_effect_ss[9] = 1  # Set ea2 to default value
    sim_0 = ode.cal_equation(1000, drug_effect_ss, x)  # 1000 weeks for steady state
    init_cond2 = sim_0[-1, :]  # Take the last state as steady state

    # Simulate with drug effects
    ode = ODE(diff_eq, init_cond2)
    T_end = 24  # weeks
    sim = ode.cal_equation(T_end, de, x)

    # Compute EASI scores
    easi_scores = EASI(sim)
    return easi_scores


def test_drug_efficacy(drug_effects=None, n_patients: int = 1000) -> dict:
    """
    Simulate the effects of drugs based on specified drug effects.

    Parameters:
    - drug_effects: dict, mapping biological factors to their effects
        The effects are specified as fractions representing the reduction (or increase) in the factor.
        For example, {"IL-4": 0.5} represents a 50% reduction in IL-4 levels.
        For additive effects (e.g., IFNg), specify the amount to be added.

    - n_patients: int, number of virtual patients to simulate

    Returns:
    - results: dict, containing mean_easi, std_easi, and easi_scores (array of individual scores)
    """

    if drug_effects is None:
        drug_effects = default_drug_effects

    # Load parameters
    mu, sigma = load_parameters()

    # Generate virtual patients
    random_list = np.random.randn(51, n_patients)
    virtual_subjects = random_list * np.abs(sigma) + mu
    virtual_subjects = np.exp(virtual_subjects)

    # Map the drug_effects dictionary to the de array
    de = np.zeros(10, dtype="float64")
    de[0] = 1e20  # As per the model, de[0] is set to a large number

    # Map the keys in drug_effects to de indices
    factor_to_de_index = {
        "IL-4": 1,
        "IL-13": 2,
        "IL-17": 3,
        "IL-22": 4,
        "IL-31": 5,
        "TSLP": 6,
        "OX40L": 7,
        "IFNg": 8,  # additive effect
        "ea2": 9,  # Used in IL13 effect
    }

    # For each factor in drug_effects, set de appropriately
    for factor, effect in drug_effects.items():
        if factor in factor_to_de_index:
            idx = factor_to_de_index[factor]
            de[idx] = effect
        else:
            pass  # Ignore factors that don't map to de

    # Set default value for ea2 if not specified
    if de[9] == 0:
        de[9] = 0
    else:
        de[9] = 1

    # Simulate for each virtual patient
    res_i = np.zeros((n_patients, 241))  # 241 time points (0 to 24 weeks, dt=0.1)

    for i in range(n_patients):
        x = virtual_subjects[:, i]
        easi_scores = simulate_patient(x, de)
        res_i[i, :] = easi_scores

    # Compute mean and std of EASI scores over patients at each time point
    mean_easi = np.mean(res_i, axis=0)
    std_easi = np.std(res_i, axis=0)

    results = {
        "mean_easi": mean_easi,
        "std_easi": std_easi,
        "easi_scores": res_i,
    }

    return results


def get_easi_severity(easi_score):
    """
    Determines the severity of eczema based on the EASI score.

    Parameters:
    - easi_score (float): The EASI score ranging from 0 to 72.

    Returns:
    - severity (str): The severity classification.
    """
    if easi_score < 0 or easi_score > 72:
        return "Invalid EASI score. It should be between 0 and 72."

    if easi_score == 0:
        severity = "Clear Skin"
    elif 0 < easi_score <= 1.0:
        severity = "Almost Clear"
    elif 1.1 <= easi_score <= 7.0:
        severity = "Mild Eczema"
    elif 7.1 <= easi_score <= 21.0:
        severity = "Moderate Eczema"
    elif 21.1 <= easi_score <= 50.0:
        severity = "Severe Eczema"
    elif 50.1 <= easi_score <= 72.0:
        severity = "Very Severe Eczema"
    else:
        severity = "Unknown Severity"

    return severity


question_examples = [
    "The drug I have increases Th2 and reduces IL-22 slightly. IFNg reduced to -0.5.",
    "This treatment significantly reduces IL-4 and slightly increases skin barrier integrity.",
    "My drug boosts Th1 significantly but decreases IL-17 and TSLP moderately.",
    "This drug lowers infiltrated pathogens, boosts Th22, and slightly reduces IL-31.",
    "The treatment decreases both IL-4 and IL-22 by 30% but boosts IL-17 significantly.",
    "The new drug increases OX40L moderately while reducing Th2 and Th17.",
    "It has a large effect on reducing Th1 and increases skin barrier integrity slightly.",
    "This drug decreases IL-13 slightly but causes a significant drop in IFNg and Th22.",
    "The medication reduces IL-31 by 50%, increases Th17 slightly, and boosts TSLP moderately.",
    "This drug greatly improves skin barrier integrity and reduces IL-4 while keeping other factors unchanged.",
]

if __name__ == "__main__":
    # Example usage: simulate drug reducing IL-4 by 50% and IL-13 by 70%
    drug_effects = {"IL-4": 0.5, "IL-13": 0.7}
    results = test_drug_efficacy(drug_effects, n_patients=1000)

    # Print the results
    print("Test Results:")
    print(f"Mean EASI Score at Week 24: {results['mean_easi'][-1]:.4f}")
    print(f"Standard Deviation of EASI Score at Week 24: {results['std_easi'][-1]:.4f}")
