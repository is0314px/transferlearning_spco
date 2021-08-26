import numpy as np
import sys 
import math
from sklearn.preprocessing import normalize
outputdir=sys.argv[1]
for dataset in range(4):
    parameter_dir=outputdir+"/dataset"+repr(dataset)+"/"
    print  parameter_dir
    try:
        phi_n=np.loadtxt(parameter_dir+"/phi_w_e.csv")
    except IOError:
        phi_n=np.loadtxt(parameter_dir+"/../phi_w.csv")

    G=np.loadtxt(parameter_dir+"/G.txt")



    prob_n=[0.0 for i in range(len(phi_n[0]))]
    for w in range(len(phi_n[0])):
        for c in range(len(G)):   
            prob_n[w] += phi_n[c][w]*G[c]
    phi_sum=np.sum(phi_n,axis=0)
    phi_sum=phi_sum/sum(phi_sum)
    #print phi_sum
    #
    """
    for c in range(len(phi_n)):
        prob_n_c[0]=normalize(prob_n_c[0])
    """


    info_n=[[0.0 for i in range(len(phi_n[0]))] for j in range(len(phi_n))]
    for c in range(len(phi_n)):
        for w in range(len(phi_n[0])):
            t_t_prob=(phi_n[c][w]*G[c])/(prob_n[w]*G[c])
            t_t_prob=(phi_n[c][w]*G[c])*math.log(t_t_prob)

            t_f_prob=((1.0-phi_n[c][w])*G[c])/((1.0-prob_n[w])*G[c])
            #print phi_sum[w]
            t_f_prob=((1.0-phi_n[c][w])*G[c])*math.log(t_f_prob)

            f_t_prob=((1-G[c])*(1.0-phi_n[c][w]))/(prob_n[w]*(1-G[c]))
            f_t_prob=((1-G[c])*(1.0-phi_n[c][w]))*math.log(f_t_prob)

            f_f_prob=((1.0-G[c])*(1.0-phi_n[c][w]))/((1.0-prob_n[w])*(1.0-G[c]))
            f_f_prob=((1.0-G[c])*(1.0-phi_n[c][w]))*math.log(f_f_prob)
            
            info_n[c][w]=t_t_prob+t_f_prob+f_t_prob+f_f_prob
        #print info_n[c][2]    
    #print info_n[0]    
    #info_n= np.exp(info_n)  

            
    #info_n=np.exp(info_n)
    """
    for n in range(len(phi_n[0])):
        sum_p=sum(info_n[n])
        info_n[n]/=sum_p
    """
    np.savetxt(parameter_dir+"/mutual_info_function.csv",info_n)