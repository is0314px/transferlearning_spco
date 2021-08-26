#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np 
import argparse
import os
import sys
import math
from numpy.linalg import inv, cholesky
import glob
import re
import shutil
from numpy.random import *
#import pypr.clustering.gmm as gmm

parameter_dir = sys.argv[1] #test dataset directory
env_num = sys.argv[2] #test dataset index (test data number) example:if env_num_0, this is 0
dic = sys.argv[3]

name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic, delimiter="\n", dtype='S' )
dir = "../gibbs_dataset/sigverse/3LDK_9" 
name_data_dir = "/name_label/"
position_data_dir = "/position/"
name_dir = dir + name_data_dir
position_dir = dir + position_data_dir

N = 20
DATA_NUM = 200
dist_num = 3 #XX's-roomは３つ

position_list = [[] for w in range(dist_num)]

#=========== make label position ==========
for i in range(DATA_NUM):

    name_file = name_dir + "word" + repr(i) + ".txt"
    name = np.loadtxt(name_file, delimiter=" ", dtype='S')
    name_list = name.tolist()

    position_file = position_dir + repr(i) + ".txt"
    position = []

    for line in open(position_file, 'r').readlines():
        try: #for similar
            data=line[:-1].split('\t')
            position += [float(data[0].replace('\xef\xbb\xbf', ''))]
    
        except: #for realworld
            data=line[:-1].split(' ')
            position +=[float(data[0].replace('\r', ''))]

        position += [float(data[1].replace('\r', ''))]
    
    position = [position[0]*100,position[1]*100,position[2],position[3]]
    
    if 20 <= i and i <= 39:
        position_list[0].append(position)
    if 40 <= i and i  <= 59:
        position_list[1].append(position)
    if 180 <= i and i <= 199:
        position_list[2].append(position)
    
label_pos = []
for idx in range(dist_num):
    x_ave = 0
    y_ave = 0
    
    for m in range(N):
        x_ave = x_ave + position_list[idx][m][0]
        y_ave = y_ave + position_list[idx][m][1]
    
    x_ave = x_ave / N
    y_ave = y_ave / N
    label_pos.append([x_ave,y_ave])

def Multi_prob(data,phi): #log p(*|phi^*)
    phi += 1e-300
    #phi_log = np.log(phi)
    phi_log = phi
    prob = data.dot(phi_log.T)
    return prob

def predict_position(name_vector,phi_n,pi,G):
    prob_rt = [0.0 for i in xrange(len(pi[0]))]

    for r in xrange(len(pi[0])):
        for c in xrange(len(G)):
            prob = Multi_prob(name_vector,phi_n[c])*G[c]*pi[c][r] #p(n_t|phi^n_c)*p(c|G)*p(r|pi_c)
            prob_rt[r] += prob

    R_t = np.argmax(prob_rt) #choose R_t

    return R_t

def caluculate_cost(idx,R_t,label_pos,mu_set):
    cost = ((label_pos[idx][0] - mu_set[R_t][0])**2) + ((label_pos[idx][1] - mu_set[R_t][1])**2) #x[0]:axis-X, x[1]:axis-Z
    cost = cost**0.5 #caluculate cost

    return cost

def make_name_vector(name):
    name_vector = []
    for nd in name_dic:
        if name == nd:
            name_vector.append(1)
        else:
            name_vector.append(0)
    
    name_vector = np.array(name_vector)

    #print(name_vector)
    
    return name_vector

per = 0
PER = 20
MAX_PER = 100

while per <= MAX_PER:
    
    per_dir = '/per_'+repr(int(per))+'_iter_200/dataset'+repr(int(env_num))

    #========== read estimated parameters of test data ==========
    try:
        phi_n = np.loadtxt(parameter_dir+per_dir+"/theta_n.csv")

    except IOError:
        phi_n = np.loadtxt(parameter_dir+per_dir+"/../phi_n.csv")

    pi = np.loadtxt(parameter_dir+per_dir+"/pi.csv")
    G = np.loadtxt(parameter_dir+per_dir+"/G.txt")
    region_num = len(pi[0])
    sigma_set = []
    mu_set = []

    for i in range(region_num):
        sigma = np.loadtxt(parameter_dir+per_dir+"/sigma/gauss_sigma"+repr(i)+".csv")
        mu = np.loadtxt(parameter_dir+per_dir+"/mu/gauss_mu"+repr(i)+".csv")
        sigma_set.append(sigma)
        mu_set.append(mu)

    #========== predict position and evaluate cost ==========
    sum = 0

    Names = []
    for i, nd in enumerate(name_dic):
        if i >= 15: #"*'s-room"が名前辞書で15番目以降であるため
            Names.append(nd)
        #elif i == 9: #bedroom
            #Names.append(nd)

    label_idx = [0,1,2] #index of name label

    N = 0
    cnt_vector = [0 for i in range(len(name_dic))]
    cost_vector = [0 for i in range(len(name_dic))]

    for i, name in enumerate(Names):
        name_vector = make_name_vector(name)
        idx = label_idx[i]

        prob_rt = [0.0 for i in xrange(len(pi[0]))]

        for r in xrange(len(pi[0])):
            for c in xrange(len(G)):
                prob = Multi_prob(name_vector,phi_n[c])*G[c]*pi[c][r] #p(n_t|phi^n_c)*p(c|G)*p(r|pi_c)
                prob_rt[r] += prob

        R_t = np.argmax(prob_rt) #choose R_t
        Mu = [mu_set[R_t][0],mu_set[R_t][1]]
        Sigma = sigma_set[R_t][:2,:2]
        sample_pos = multivariate_normal(Mu, Sigma, 10)

        for pos in sample_pos:
            x = pos[0]
            y = pos[1]

            if name == "Emma's-room":
                cost = (((label_pos[idx][0] - x)**2) + ((label_pos[idx][1] - y)**2)) **0.5
                cost_vector[name_dic.tolist().index(name)] += cost

                if (-590 <= x and x <= -50) and (-470 <= y and y <= -160):
                    cnt_vector[name_dic.tolist().index(name)] += 1

            elif name == "mother's-room":
                cost = (((label_pos[idx][0] - x)**2) + ((label_pos[idx][1] - y)**2)) **0.5
                cost_vector[name_dic.tolist().index(name)] += cost

                if (-590 <= x and x <= -200) and (-160 <= y and y <= 330):
                    cnt_vector[name_dic.tolist().index(name)] += 1

            elif name == "father's-room":
                cost = (((label_pos[idx][0] - x)**2) + ((label_pos[idx][1] - y)**2)) **0.5
                cost_vector[name_dic.tolist().index(name)] += cost

                if ((100 <= x and x <= 150) and (-350 <= y and y <= -160)) or ((150 <= x and x <= 590) and (-480 <= y and y <= -160)):
                    cnt_vector[name_dic.tolist().index(name)] += 1

            N += 1

    print("position accuracy")
    print("name given rate: "+repr(per))

    max_width = max(len(s) for s in name_dic)

    for n, name in enumerate(name_dic):
        if n >= 15: #Emma's-room ~ 
            acc_each = float(cnt_vector[n])/len(sample_pos)
            print(name+" "*(max_width-len(name_dic[n]))+": "+repr(np.round(acc_each,2)))
            f = open(parameter_dir+'/Position_evaluation/pos_acc_'+name+'.txt','a')
            f.write(repr(per)+" "+repr(acc_each)+'\n')
            f.close()
    
    acc_ave = float(np.sum(cnt_vector)) / float(N)
    print("acc_ave"+" "*(max_width-7)+": "+repr(np.round(acc_ave,2)))
    file = open(parameter_dir+'/Position_evaluation/pos_acc_specific.txt','a')
    file.write(repr(per)+" "+repr(acc_ave)+'\n')
    file.close()

    print("position cost")
    print("name given rate: "+repr(per))

    for n, name in enumerate(name_dic):
        if n >= 15: #Emma's-room ~ 
            cost_each = float(cost_vector[n])/len(sample_pos)
            print(name+" "*(max_width-len(name_dic[n]))+": "+repr(np.round(cost_each,2)))
            f = open(parameter_dir+'/Position_evaluation/pos_cost_'+name+'.txt','a')
            f.write(repr(per)+" "+repr(cost_each)+'\n')
            f.close()

    cost_ave = float(np.sum(cost_vector)) / float(N)
    print("cost_ave"+" "*(max_width-8)+": "+repr(np.round(cost_ave,2)))
    file = open(parameter_dir+'/Position_evaluation/pos_cost_specific.txt','a')
    file.write(repr(per)+" "+repr(cost_ave)+'\n')
    file.close()
    
    per = per + PER
