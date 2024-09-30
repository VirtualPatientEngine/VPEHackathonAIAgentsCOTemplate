# -*- coding: utf-8 -*-
"""
This software is released under the MIT License, see LICENSE.txt.
Copyright (c) 2021 Takuya Miyano

QSP model of S. aureus-targeted therapies simulates %improved EASI, EASI-75 and other biological factors

"""
import numpy as np
from scipy.integrate import odeint
from numba import jit

def simulate(mu, sigma, n_patients):
    """
    Parameters
    ----------
    mu      : ndarray of shape (26, 1)
        26 distribution parameters that represent mean of ln(k_i)
    
    sigma   : ndarray of shape (26, 1)
        26 distribution parameters that represent standard deviation of ln(k_i)
    
    n_patients : int
        number of virtual patients
    
    Returns
    ----------
    mean_ss : ndarray of shape (6)
        mean of baseline levels of 6 biological factors
        EASI score, skin barrier integrity, S. aureus, CoNS, agr expression and IL4/13 
    
    cv_ss   : ndarray of shape (6)
        %CV of baseline levels of 6 biological factors
        EASI score, skin barrier integrity, S. aureus, CoNS, agr expression and IL4/13 
    
    the followings are ndarray of shape (241, 5)
        161 time points from week 0 to 16 (placebo, dupilumab and flucloxacillin) or to 1.6 weeks (ShA9-sensitive and ShA9-resistant)
        5 invterventions of placebo, dupilumab, ShA9-sensitive, ShA9-resistant, flucloxacillin
 
    impEASI : %improved EASI
    e75     : EASI-75
    mSa     : mean S. aureus level
    sdSa    : S.D. of S. aureus level
    mSo     : mean CoNS level
    sdSo    : S.D. of CoNS level
    mb      : mean skin barrier integrity  level
    sdb     : S.D. of skin barrier integrity level
    ma      : mean agr expression  level
    sda     : S.D. of agr expression level
    m4      : mean IL4/13 level
    sd4     : S.D. of IL4/13 level

   """

    class ODE(object):
        def __init__(self, diff_eq, init_con):
            self.diff_eq  = diff_eq
            self.init_con = init_con
            
        def cal_equation(self, t_end, drug_effect, x):
            dt = 0.1 # delta time (week)
            N = round(t_end/dt) + 1 #Time steps
            t = np.linspace(0, t_end, N) # prepare time
            v = odeint(self.diff_eq, self.init_con, t, rtol=1e-8, atol=1e-6, args=(drug_effect, x))
            return v
    
    class ODE2(object):
        def __init__(self, diff_eq, init_con):
            self.diff_eq  = diff_eq
            self.init_con = init_con
            
        def cal_eq2(self, t_end, drug_effect, x):
            dt = 0.01 # delta time (week)
            N = round(t_end/dt) + 1 #Time steps
            t = np.linspace(0, t_end, N) # prepare time
            v = odeint(self.diff_eq, self.init_con, t, rtol=1e-8, atol=1e-6, args=(drug_effect, x))
            return v

    @jit('f8[:](f8[:,:],f8,f8)', nopython=True)
    def EASI(sim, k1, b1):
        s = sim[:,0]
        a = sim[:,1]
        h = sim[:,2]
        a[a < 0] = 0
        h[h < 0] = 0
        a_agr = np.tanh(k1*a / (1 + b1*h))
        e = 72 * (2*a_agr + 2*(1-s)) /4
        return e
    
    @jit('f8[:](f8[:,:],f8,f8,f8)', nopython=True)
    def EASI_A9(sim, k1, b1, bA9a):
        s = sim[:,0]
        a = sim[:,1]
        h = sim[:,2]
        a[a < 0] = 0
        h[h < 0] = 0
        a_agr = np.tanh(k1*a / ((1 + b1*h) * (1 + bA9a)))
        e = 72 * (2*a_agr + 2*(1-s)) /4
        return e

    @jit('f8[:](f8[:],f8,f8[:],f8[:])', nopython=True, debug=True)
    def diff_eq(c, t, de, x):
    # c : levels of 4 biological factors at t. ndarray of shape (4)
    #    c[0] : skin barrier integrity
    #    c[1] : S. aureus
    #    c[2] : CoNS
    #    c[3] : IL4/13
    # t : time (int)
    # de: drug effect [placebo, IL4, ShA9s, ShA9r, flu]
    # x : 26 parameter values 
        k1 = x[0]
        k2 = x[1] 
        k3 = min([x[2],de[0]]) # placebo effect on decrease in skin barrier
        k4 = x[3] 
        k5 = x[4]
        k6 = x[5]
        k7 = x[6]
        b1 = x[7]
        b2 = x[8]
        b3 = x[9]
        b4 = x[10]
        d1 = x[11]
        d2 = x[12]
        d3 = x[13]
        d4 = x[14]
        d5 = x[15]
        d6 = x[16]
        d7 = x[17]
        d8 = x[18]
        d9 = x[19]
        dA9as = min([x[20], de[2]])
        dA9ar = min([x[21], de[3]])
        dA9h = min([x[22], max([de[2], de[3]])])
        bA9a = min([x[23], max([de[2], de[3]])])
        dfa = min([x[24], de[4]])
        dfh = min([x[25], de[4]])
        a_max = np.float64(7)
        h_max = np.float64(7)
        c[1] = max([c[1],0])
        c[2] = max([c[2],0])

        IL4  = (1 - de[1])*c[3]
        a_agr = np.tanh(k1*c[1] / ((1 + b1*c[2]) * (1 + bA9a)))

        #   ODEs
        dc0dt = (1 - c[0])*(k2 + k3)/(1 + b2*IL4)  - c[0]*(d1 + d2*a_agr)
        dc1dt = k4* (1 - c[1]/a_max)/(1 + b3*c[0]) - (d3*c[2] + d4/(1 + b4*IL4) + d5 + dA9as + dA9ar + dfa)
        dc2dt = k5* (1 - c[2]/h_max)               - (d6*c[1] + d7/(1 + b4*IL4) + d8 + dA9h + dfh)
        dc3dt = k6*a_agr + k7 - d9*c[3]  # IL4
        
        results_all = np.array([dc0dt, dc1dt, dc2dt, dc3dt])
        return results_all

    def simulate_Tend(x):    
        # simulate steady state (1000 weeks) /baseline levels
        # initial conditions for simulating steady-state levels of biological factors 
        s_0    = np.float64(0.5931) # tentative
        a_0    = np.float64(7) 
        h_0    = np.float64(7)
        IL4_0  = np.float64(39.2)
        init_cond = np.array([s_0, a_0, h_0, IL4_0], dtype='float64')

        ode = ODE(diff_eq, init_cond)
        # de: drug effect [placebo, IL4, ShA9s, ShA9r, flu]
        sim_0 = ode.cal_equation(1000, np.array([0, 0, 0, 0, 0], dtype='float64'), x) # 1000 days for steady state
        init_cond2 = sim_0[10000,:]

        # use steady-state level as baseline levels (initial condition)
        ode = ODE(diff_eq, init_cond2)
        T_end = 16 # weeks                         plac   IL4   A9s   A9r   flu
        sim_1 =  ode.cal_equation(T_end, np.array([1E20,    0,    0,    0,    0], dtype='float64'), x) # Placebo (other)
        sim_2 =  ode.cal_equation(T_end, np.array([1E20, 0.99,    0,    0,    0], dtype='float64'), x) # Dupilumab (IL4/13)
    
        ode2 = ODE2(diff_eq, init_cond2)
        sim_3 = ode2.cal_eq2(1.6, np.array([1E20,    0,    1E20,    0,    0], dtype='float64'), x) # ShA9 sensitive
        sim_4 = ode2.cal_eq2(1.6, np.array([1E20,    0,       0, 1E20,    0], dtype='float64'), x) # ShA9 resistant
    
        sim_5a = ode.cal_equation(4, np.array([1E20,    0,    0   , 0,   1E20], dtype='float64'), x) # flucloxacillin
        init_cond5 = sim_5a[40,:]
        ode = ODE(diff_eq, init_cond5)
        sim_5b = ode.cal_equation(T_end - 4.1, np.array([1E20,    0,    0   , 0,    0], dtype='float64'), x) # flucloxacillin
        sim_5 = np.concatenate([sim_5a,sim_5b])
        return sim_0, sim_1, sim_2, sim_3, sim_4, sim_5

    def agr_A9(sim, x):
        k1 = x[0]
        b1 = x[7]
        bA9a = x[23]
        a = sim[:,1]
        h = sim[:,2]
        a[a < 0] = 0
        h[h < 0] = 0
        a_agr = np.tanh(k1*a / ((1 + b1*h) * (1 + bA9a)))
        return a_agr
    
    def agr(sim, x):
        k1 = x[0]
        b1 = x[7]
        a = sim[:,1]
        h = sim[:,2]
        a[a < 0] = 0
        h[h < 0] = 0
        a_agr = np.tanh(k1*a / (1 + b1*h))
        return a_agr
    
    def results_all(x):    
        k1 = x[0]
        b1 = x[7]
        bA9a = x[23]
        sim_0, sim_1, sim_2, sim_3, sim_4, sim_5 = simulate_Tend(x)
    
        a_agr1 = agr(sim_1, x)
        a_agr2 = agr(sim_2, x)
        a_agr3 = agr_A9(sim_3, x)
        a_agr4 = agr_A9(sim_4, x)
        a_agr5 = agr(sim_5, x)
    
        b = np.concatenate([np.array([EASI(sim_0, k1, b1)[10000]]), sim_0[10000,0:4]])
        Res_series = np.concatenate([b, \
                                  EASI(sim_1, k1, b1), EASI(sim_2, k1, b1), \
                                  EASI_A9(sim_3, k1, b1, bA9a),\
                                  EASI_A9(sim_4, k1, b1, bA9a),\
                                  EASI(sim_5, k1, b1),\
                                  sim_1[:,0], sim_2[:,0], sim_3[:,0], sim_4[:,0], sim_5[:,0],\
                                  sim_1[:,1], sim_2[:,1], sim_3[:,1], sim_4[:,1], sim_5[:,1],\
                                  sim_1[:,2], sim_2[:,2], sim_3[:,2], sim_4[:,2], sim_5[:,2],\
                                  a_agr1,    a_agr2,     a_agr3,     a_agr4,     a_agr5,\
                                  sim_1[:,3], sim_2[:,3]*0.01, sim_3[:,3], sim_4[:,3], sim_5[:,3]])
        return Res_series

    # prepare virtual patients
    n_patients_eval = 1000
    random_list = np.random.randn(26, n_patients)
    virtual_subjects = random_list*sigma + mu
    virtual_subjects = np.exp(virtual_subjects)
    sampleList = [(i, virtual_subjects) for i in range(n_patients_eval)]

    # simulation using the virtual patients
    res_i = np.zeros([n_patients_eval,4835])
    for i in range(n_patients_eval):
        res_i[i] = results_all(sampleList[i][1][:,i])
    a = res_i.T
    a = a[:, np.all(a[5:810,:] < 72, axis=0)]
    
    mean_ss = np.mean(a[:5,:], axis = 1).T.reshape(-1,5).T
    cv_ss   = 100*np.std(a[:5,:], axis = 1)/np.mean(a[:5,:], axis = 1)
    cv_ss   = cv_ss.T.reshape(-1,5).T
    impEASI = 100*(a[0,:].reshape(1,-1) - a[5:810,:])/a[0,:].reshape(1,-1)
    e75     = np.count_nonzero(impEASI > 75, axis=1)/n_patients_eval*100
    e75     = e75.T.reshape(-1,161).T
    mb      = np.mean(a[810:1615,:], axis = 1).T.reshape(-1,161).T
    sdb     = np.std(a[810:1615,:], axis = 1).T.reshape(-1,161).T
    mSa     = np.mean(a[1615:2420,:], axis = 1).T.reshape(-1,161).T
    sdSa    = np.std(a[1615:2420,:], axis = 1).T.reshape(-1,161).T
    mSo     = np.mean(a[2420:3225,:], axis = 1).T.reshape(-1,161).T
    sdSo    = np.std(a[2420:3225,:], axis = 1).T.reshape(-1,161).T
    ma      = np.mean(a[3225:4030,:], axis = 1).T.reshape(-1,161).T
    sda     = np.std(a[3225:4030,:], axis = 1).T.reshape(-1,161).T
    m4      = np.mean(a[4030:4835,:], axis = 1).T.reshape(-1,161).T
    sd4     = np.std(a[4030:4835,:], axis = 1).T.reshape(-1,161).T

    return mean_ss, cv_ss, impEASI, e75, mSa, sdSa, mSo, sdSo, mb, sdb, ma, sda, m4, sd4

if __name__ == "__main__":
    mu = np.loadtxt("mu_Saureus.csv", delimiter = ",", dtype = float).reshape(-1,1)
    sigma = np.loadtxt("sigma_Saureus.csv", delimiter = ",", dtype = float).reshape(-1,1)
    n_patients = 1000
    mean_ss, cv_ss, impEASI, e75, mSa, sdSa, mSo, sdSo, mb, sdb, ma, sda, m4, sd4 = simulate(mu, sigma, n_patients)
