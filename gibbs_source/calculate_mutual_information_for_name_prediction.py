#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import os
import sys 
import math
#from sklearn.preprocessing import normalize
from numpy.linalg import inv, cholesky
import scipy.stats as ss
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    from sklearn.preprocessing import normalize

output_path = sys.argv[1] # mutual_path_*.txt 
training_dataset_index = int(sys.argv[2]) #index of test dataset  #if mutual_path_*.txt, this is *

outputdir_list = np.loadtxt(output_path, delimiter="\n", dtype="S",ndmin=1) 

for outputdir in outputdir_list:

    parameter_dir = outputdir+"/dataset"+repr(training_dataset_index)

    #=============== read parameters ===============
    try:
        phi_n = np.loadtxt(parameter_dir+"/theta_n.csv")
    except IOError:
        phi_n = np.loadtxt(outputdir+"/phi_n.csv")

    G = np.loadtxt(parameter_dir+"/G.txt")

    #=============== calculate P(n|Theta),n = {n_t,n_t_ber} Theta = {phi_n,G} ===============
    prob_n = [0.0 for i in range(len(phi_n[0]))]

    for w in range(len(phi_n[0])):
        for c in range(len(G)):   
            prob_n[w] += phi_n[c][w] * G[c]
    
    #=============== calucurate mutual information ===============
    mutual_info = [[0.0 for i in range(len(phi_n[0]))] for j in range(len(phi_n))] #mutual information

    for c in range(len(phi_n)):
        for w in range(len(phi_n[0])):
            t_t_prob = (phi_n[c][w] * G[c]) / (prob_n[w] * G[c])
            t_t_prob = (phi_n[c][w] * G[c]) * math.log(t_t_prob) #P(n_t,C_t|Theta) * log()
            
            t_f_prob = ((1.0-phi_n[c][w]) * G[c]) / ((1.0 - prob_n[w]) * G[c])
            if t_f_prob == 0.0:
                t_f_prob = 1e-100
                #t_f_prob = 1.0
            t_f_prob = ((1.0-phi_n[c][w]) * G[c]) * math.log(t_f_prob) #P(n_t_ber,C_t|Thera) * log()
            
            f_t_prob = ((1-G[c]) * phi_n[c][w]) / (prob_n[w] * (1.0-G[c]))
            if f_t_prob == 0.0:
                f_t_prob = 1e-100
                #f_t_prob = 1.0
            f_t_prob = ((1-G[c]) * phi_n[c][w]) * math.log(f_t_prob) #P(n_t,C_t_ber|Theta) * log()
            
            f_f_prob = ((1.0-G[c]) * (1.0-phi_n[c][w])) / ((1.0-prob_n[w]) * (1.0-G[c]))
            if f_f_prob == 0.0:
                f_f_prob = 1e-100
                #f_f_prob = 1.0
            f_f_prob = ((1.0-G[c]) * (1.0-phi_n[c][w])) * math.log(f_f_prob) #P(n_t_ber,C_t_ber|Theta) * log()
        
            mutual_info[c][w] = t_t_prob + t_f_prob + f_t_prob + f_f_prob 

    np.savetxt(parameter_dir+"/mutual_info.csv", mutual_info)
  

