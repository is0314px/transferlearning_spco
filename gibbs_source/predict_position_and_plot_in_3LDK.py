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
import matplotlib.pylab as plt
from PIL import Image
#import pypr.clustering.gmm as gmm

parameter_dir = sys.argv[1] #test dataset directory
env_num = sys.argv[2] #test dataset index (test data number) example:if env_num_0, this is 0
dic = sys.argv[3]

name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic, delimiter="\n", dtype='S' )
#dir = "../gibbs_dataset/sigverse/3LDK_9" 
dir = "../gibbs_dataset/sigverse_subject/3LDK_9" 
position_data_dir = "/position/"
position_dir = dir + position_data_dir

N = 20
DATA_NUM = 200

def Multi_prob(data,phi): #log p(*|phi^*)
    phi += 1e-300
    phi_log = phi
    prob = data.dot(phi_log.T)
    return prob

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
MAX_PER = 0

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
    Names = []
    """
    for i, nd in enumerate(name_dic):
        if i > 6: #"*'s-room"が名前辞書で14番目以降であるため
            Names.append(nd)
    """
    #Names = ["door","restroom","bedroom","bath","living","dining","fridge","washroom"] #"Emma's-room","mother's-room","father's-room"]
    Names = ["entrance","toilet","bedroom","bath","living-room","dining","kitchen","washroom","Emma's-room","mother's-room","father's-room"]
    #Names = ["door","toilet","bedroom","bathroom","living-room","table","refrigerator","sink","Emma's-room","mother's-room","father's-room"]

    print("-------------------------")
    print("per:"+repr(per))

    Spx = []
    Spy = []

    for i, name in enumerate(Names):
        spx = []
        spy =[]

        name_vector = make_name_vector(name)

        prob_rt = [0.0 for i in xrange(len(pi[0]))]

        for r in xrange(len(pi[0])):
            for c in xrange(len(G)):
                prob = Multi_prob(name_vector,phi_n[c])*G[c]*pi[c][r] #p(n_t|phi^n_c)*p(c|G)*p(r|pi_c)
                prob_rt[r] += prob

        R_t = np.argmax(prob_rt) #choose R_t
        #pre_pos = [mu_set[R_t][0],mu_set[R_t][1]]
        Mu = [mu_set[R_t][0],mu_set[R_t][1]]
        Sigma = sigma_set[R_t][:2,:2]

        #print(sigma_set[R_t])
        #print(Sigma)
        sample_pos = multivariate_normal(Mu, Sigma, 10)
        #print(sample_pos)

        for j in range(len(sample_pos)):
            spx.append(sample_pos[j][0])
            spy.append(sample_pos[j][1])
        
        Spx.append(spx)
        Spy.append(spy)

    fig, ax = plt.subplots()

    """
    plt.scatter(Spx[8],Spy[8],edgecolor = 'black',facecolor = 'blue',marker = 'o',s = 50,label="Emma's-room",alpha=0.6)
    plt.scatter(Spx[9],Spy[9],edgecolor = 'black',facecolor = 'red',marker = 'X',s = 50,label="mother's-room",alpha=0.6)
    plt.scatter(Spx[10],Spy[10],edgecolor = 'black',facecolor = 'green',marker = '*',s = 50,label="father's-room",alpha=0.6)
    #"""
    #"""
    plt.scatter(Spx[0],Spy[0],edgecolor = 'black',facecolor = 'cyan',marker = '^',s = 50,label=Names[0],alpha=0.6)
    plt.scatter(Spx[1],Spy[1],edgecolor = 'black',facecolor = 'magenta',marker = 'v',s = 50,label=Names[1],alpha=0.6)
    plt.scatter(Spx[3],Spy[3],edgecolor = 'black',facecolor = 'yellow',marker = 'd',s = 50,label=Names[3],alpha=0.6)
    plt.scatter(Spx[4],Spy[4],edgecolor = 'black',facecolor = 'mediumspringgreen',marker = 's',s = 50,label=Names[4],alpha=0.6)
    plt.scatter(Spx[5],Spy[5],edgecolor = 'black',facecolor = 'black',marker = 'p',s = 50,label=Names[5],alpha=0.6)
    plt.scatter(Spx[6],Spy[6],edgecolor = 'black',facecolor = 'navy',marker = 'h',s = 50,label=Names[6],alpha=0.6)
    plt.scatter(Spx[7],Spy[7],edgecolor = 'black',facecolor = 'orangered',marker = 'P',s = 50,label=Names[7],alpha=0.6)
    plt.scatter(Spx[2],Spy[2],edgecolor = 'black',facecolor = 'white',marker = 'D',s = 50,label=Names[2],alpha=0.6)
    #"""
    
    x_min = -590
    x_max =  590
    y_min = -470
    y_max =  470

    img = Image.open("map/3ldk_9_local.PNG")
    plt.imshow(img,extent=(x_min,x_max,y_min,y_max))

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    plt.xlabel('x',fontsize=14)
    plt.ylabel('y',fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8,ncol=4)
    #plt.legend(loc=(0, 0.7), borderaxespad=0, fontsize=8,ncol=2)
    plt.savefig("gibbs_result/sigverse_result/Position_prediction_result/plotted_map/plotted_predicted_position.png",dpi=1000)
                
    per = per + PER
