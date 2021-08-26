#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np 
import os
import sys
import math
from numpy.linalg import inv, cholesky
import scipy.stats as ss
import matplotlib.pylab as plt

train_list = sys.argv[1] 
result_dir = sys.argv[2]
dic_txt = sys.argv[3]
ex_type = sys.argv[4]

name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic_txt, delimiter="\n", dtype='S' )

def Multi_prob(data,phi): #log p(*|phi^*)
    phi += 1e-300
    phi_log = np.log(phi)
    prob = data.dot(phi_log.T)
    return prob

def normalize(probs): 
    prob_factor = 1.0 / sum(probs)
    return [prob_factor * p for p in probs]

def Name_data_read(file,name_increment): #read name data
    name_data = [0 for w in range(len(name_dic))]    
    data = np.loadtxt(file, delimiter=' ', dtype='S' )
    data_l = data.tolist()
    
    if isinstance(data_l, list) == False:
        for w,dictionry in enumerate(name_dic):
            if data_l == dictionry:
                name_data[w] += name_increment
    else:
        for d in data_l:
            for w,dictionry in enumerate(name_dic):   
                if d == dictionry:
                    name_data[w] += name_increment
                
    name_data = np.array(name_data)
    return name_data

def test_data_read(file,test_num): #read number of test data
    i = 0
    test_data_num = []

    for line in open(file, 'r').readlines(): # 0,1,2,3,8,9,...
        if i == test_num: # line's row number == test_num
            num=line[:-1].split(',') # make number list and delete \n  [0,1,2,3,8,9]
            for n in num:
                try:
                    test_data_num.append(int(n))
                except ValueError:
                    pass
        i+=1
    
    return test_data_num

Training_list = np.genfromtxt(train_list , delimiter="\n", dtype='S' )

try:
    test_dataset_index = len(Training_list) - 1
except:
    test_dataset_index = 0

mute = "on" #you can choose mutual infomation is on or off
test_num = 0 #python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_similar/path_num${e}_D.txt gibbs_result/similar_result/env_num_$e transfer_similar/name_dic_local.txt
best = 1 #python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_similar/path_num${e}_D.txt gibbs_result/similar_result/env_num_$e transfer_similar/name_dic_local.txt
#out_put = "name_estimate_value.txt" #Check file name!
out_put = "predicted_names.txt"

per = 0
PER = 20

if ex_type == "general":
    MAX_PER = 0
elif ex_type == "specific":
    MAX_PER = 100
else:
    print("ERROR")
    sys.exit()

try:
    data_set_num=len(Training_list) #number of trained rooms
except TypeError:
    Training_list=np.array([Training_list])
    data_set_num=1

while per <= MAX_PER:
    #========== Read new environment parameter and test data ==========
    parameter_dir = result_dir+"/per_"+repr(per)+"_iter_200/dataset"+repr(int(test_dataset_index))

    pi = np.loadtxt(parameter_dir+"/pi.csv") #read pi
    G = np.loadtxt(parameter_dir+"/G.txt") #read G
    mutual_info = np.loadtxt(parameter_dir+"/mutual_info.csv") #read mutual infomation I(n_t,C_t|Theta)

    posdist_num = len(pi[0])
    sigma_set = []
    mu_set = []
    posdist_count = np.loadtxt(parameter_dir+"/posdist_count.txt")

    for i in range(posdist_num):
        sigma = np.loadtxt(parameter_dir+"/sigma/gauss_sigma"+repr(i)+".csv") #read sigma
        mu = np.loadtxt(parameter_dir+"/mu/gauss_mu"+repr(i)+".csv") #read mu
        sigma_set.append(sigma)
        mu_set.append(mu)
    
    try:
        phi_n = np.loadtxt(parameter_dir+"/theta_n.csv")
        phi_v = np.loadtxt(parameter_dir+"/theta_v.csv")
    except:
        try:
            phi_n = np.loadtxt(parameter_dir+"/../phi_n.csv")
            phi_v = np.loadtxt(parameter_dir+"/../phi_v.csv")
        except:
            phi_n = np.loadtxt(parameter_dir+"/theta_n.csv")

    prob_n = normalize(np.sum(phi_n,axis=0))[0] 
    f_result = open(parameter_dir+"/"+out_put,'w')
    train_dir = Training_list[int(test_dataset_index)]

    if data_set_num < len(Training_list): 
        test_file = train_dir + "/test_num_all.txt"
    else: #read test number
        test_file = train_dir + "/test_num.txt"

    test_data_num = test_data_read(test_file,test_num)

    test_vision_set = []
    test_pose_set = []
    test_name_set = []
    point_test_set = []

    for i in test_data_num:
        try:
            test_vision = np.loadtxt(train_dir+"/vision_fc7_normalized/"+repr(i)+".csv",delimiter=",") #read vision data
            test_vision_set.append(test_vision)
        except:
            pass

        f = train_dir+"/position/"+repr(i)+".txt"
        test_pose = []

        for pos in open(f, 'r').readlines(): #read position data
            
            data = pos[:-1].split('\t')
            try:
                data = pos[:-1].split('\t')
                test_pose +=[float(data[0].replace('\xef\xbb\xbf', ''))]
            except:
                data = pos[:-1].split(' ')
                test_pose +=[float(data[0].replace('\r', ''))]

            test_pose +=[float(data[1].replace('\r', ''))]

        test_pose = [test_pose[0]*100,test_pose[1]*100,test_pose[2],test_pose[3]] #x and y multiplied by 100 注意！！
        test_pose_set.append(test_pose)

    test_vision_set = np.array(test_vision_set)
    test_pose_set = np.array(test_pose_set)
    test_name_set = np.array(test_name_set)

    #========== Predict name ==========
    gauss_prob_set = np.zeros((posdist_num,len(test_data_num)),dtype=float)
    for r in range(posdist_num):
        gauss_prob = ss.multivariate_normal.logpdf(test_pose_set,mu_set[r],sigma_set[r])    
        gauss_prob_set[r] += gauss_prob

    gauss_prob_set = gauss_prob_set.T
    posdist_max = np.max(gauss_prob_set,axis=1)
    gauss_prob_set = gauss_prob_set -posdist_max[:,None]
    gauss_prob_set = np.exp(gauss_prob_set)
    sum_set = np.sum(gauss_prob_set,axis=1)
    gauss_prob_set = gauss_prob_set / sum_set[:,None]

    for i ,idx in enumerate(test_data_num):
        concept_prob = np.array([0.0 for k in range(len(pi))])
        for c in range(len(pi)):
            try:
                concept_prob[c] += Multi_prob(test_vision_set[i],phi_v[c]) # * p(v_t|phi^v_c)
                concept_prob[c] += math.log(G[c]) # * p(c|G)
            except:
                concept_prob[c] += math.log(G[c])

            posdist_prob=0.0
            for r in range(posdist_num):
                if posdist_count[r]>0:
                    posdist_prob += pi[c][r]*gauss_prob_set[i][r] # * p(r|pi_c) * p(x_t|mu_r,sigma_r)

            if posdist_prob != 0.0:
                concept_prob[c] += math.log(posdist_prob)

        max_c = np.argmax(concept_prob)    
        concept_prob -= concept_prob[max_c]
        concept_prob = np.exp(concept_prob)    
        concept_prob = normalize(concept_prob)

        prob1_1 = np.array([0.0 for k in range(len(name_dic))]) #initialize p(n_t|v_t.x_t)

        for n in range(len(name_dic)):
            for c in range(len(pi)):
                if mute == "off":
                    prob1_1[n] += concept_prob[c]*phi_n[c][n] # * p(n_t|phi^n_c)
                else:
                    prob1_1[n] += concept_prob[c]*phi_n[c][n]*mutual_info[c][n] # * p(n_t|phi^n_c) * I(n_t,C_t) #default
        
        prob1_1 = normalize(prob1_1)
        name_prob = prob1_1
        index = np.argsort(prob1_1)[::-1]
        prob1_1 = np.sort(prob1_1)[::-1]

        f_result.write(repr(test_data_num[i])+" ") #check!
        for rank in range(best):
            #f_result.write(name_dic[index[rank]]+" "+"("+repr(np.round(prob1_1[rank],3))+")"+" ") #predected word and probability
            f_result.write(name_dic[index[rank]]+" ")

        f_result.write("\n")
    f_result.close()

    per = per + PER
