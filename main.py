from src.nbse import CoupledGapEquation
from src.data_processing import DataProcessing

import numpy as np 

def main():
    # The variable kept same as in the paper draft to avoid any confusion and hence small
    # mass
    m = 1      
    # ratio of the singlet of triplet coupling         
    r = 2    
    # inter-pocket coupling, i.e., between Gamma and K pocket         
    us = 0.05        
    # interaction coupling. 
    # It has to be less than \mu, i.e. chemical potential. 
    # Therefore, it rescaled with \mu
    g = 0.1     
    # Density of states in 2D parabolic spectrum        
    n = m / (2 * np.pi) 

    # a = ratio of \xi^{-1}/2k_F
    # this following decides the range of a for plots a values and its step size
    astart = 0.0026     
    alast = 0.01          
    astep = 0.00005
    
    # this solves the SC gap matrix for Gamma pocket and multi-pocket system for the specific irreducible repreentation channel
    coupledgapequation = CoupledGapEquation(m, r, us, g, n)
    coupledgapequation.run()

    # below we save the data files to post-process to plot the phase boundaries. 
    path_pd = './data'+'/pdphasegammapkt.csv'
    data_process_pd_phase_bdd = DataProcessing(astart,alast,astep,path_pd)
    phase_sf_to_pd_bdd = data_process_pd_phase_bdd.phase_bdd()
    print(phase_sf_to_pd_bdd)
    
    # data files for all pocket included solution for the SC gap equation
    path_pd_all = './data'+ '/allpkt_wrt_pd.csv'
    data_process_pd_phase_bdd = DataProcessing(astart,alast,astep,path_pd_all)
    phase_pd_with_all_bdd = data_process_pd_phase_bdd.phase_bdd()
    print(phase_pd_with_all_bdd)
    
    #path_sf_all = './data'+ '/allpkt_wrt_sf.csv'
    #data_process_pd_phase_bdd = DataProcessing(astart,alast,astep,path_sf_all)
    #phase_sf_with_all_bdd = data_process_pd_phase_bdd.phase_bdd()
    #print(phase_sf_with_all_bdd)

if __name__ == "__main__":
    main()
