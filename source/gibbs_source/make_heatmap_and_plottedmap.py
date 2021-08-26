#!/usr/bin/env python
# -*- coding:utf-8 -*-

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
from PIL import Image
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker, cm, colors
import random

itv = 10 #plot interval, The smaller this value is, the more accurate the heat map can be generated. However it takes longer time.

result_dir = sys.argv[1]
dic_txt = sys.argv[2]
trial = sys.argv[3]

#X, Y = np.mgrid[x_min:x_max+itv:itv, y_min:y_max+itv:itv]
#Position = np.dstack((X, Y))
name_dic =np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic_txt, delimiter = "\n", dtype = "S" )
env_para = np.genfromtxt("map/3LDK_9.txt",dtype = int,delimiter = " ")
x_min = env_para[0]
x_max = env_para[1]
y_min = env_para[2]
y_max = env_para[3]
X, Y = np.mgrid[x_min:x_max+itv:itv, y_min:y_max+itv:itv]
Position = np.dstack((X, Y))
#dir = result_dir+"/spcotransfer20+MI_16"
dir = result_dir+"/spcoa"
#print(dir)

per = 0
PER = 20
MAX_PER = 100

while per <= MAX_PER:
    #parameter_dir = dir+"/per_"+repr(per)+"_iter_200/dataset16"
    parameter_dir = result_dir+"/per_"+repr(per)+"_iter_200/dataset0"

    pi = np.loadtxt(parameter_dir+"/pi.csv") #read pi
    G = np.loadtxt(parameter_dir+"/G.txt") #read G
    theta_n = np.loadtxt(parameter_dir+"/theta_n.csv")
    posdist_count = np.loadtxt(parameter_dir+"/posdist_count.txt")
    concept_count = np.loadtxt(parameter_dir+"/concept_count.txt")

    concept_num = len(G)
    posdist_num = len(pi[0])
    sigma_set = []
    mu_set = []

    for i in range(posdist_num):
        sigma = np.loadtxt(parameter_dir+"/sigma/gauss_sigma"+repr(i)+".csv") #read sigma
        mu = np.loadtxt(parameter_dir+"/mu/gauss_mu"+repr(i)+".csv") #read mu
        sigma_set.append(sigma)
        mu_set.append(mu)

    for n, name in enumerate(name_dic):
        if n >= 15 or n == 9: #X's-room, bedroom
            Probability = []
            Probability_log = []
            Coordinate = []

            for Pos in Position:
                Prob_log = []

                for pos in Pos:
                    #print(pos)
                    sum = 0
                    for c in range(concept_num):
                        if concept_count[c] > 0:
                            for r in range(posdist_num):
                                if posdist_count[r] > 0:
                                    Mu = mu_set[r][0:2]
                                    Mu = Mu.tolist()
                                    Sigma = sigma_set[r][0:2,0:2]
                                    Sigma = Sigma.tolist()
                                    Gauss = multivariate_normal(Mu, Sigma)
                                    prob = Gauss.pdf(pos) * pi[c][r] * theta_n[c][n] * G[c]
                                    #prob = np.log(prob)
                                    sum += prob
                                else:
                                    pass
                        else:
                            pass

                    Probability.append(sum)
                    Prob_log.append(np.log(sum))
                    Coordinate.append((pos[0],pos[1]))

                Probability_log.append(Prob_log)
            
            sample_positions = random.choices(Coordinate,k=100,weights=Probability)

            x=[]
            y=[]
            for pos in sample_positions:
                x.append(pos[0])
                y.append(pos[1])

            fig, ax = plt.subplots()
            plt.scatter(x,y,marker='s',c='blue',s=8,alpha=0.6)
            img = Image.open("map/3LDK_9.PNG")
            plt.imshow(img,extent=(x_min,x_max,y_min,y_max))
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            plt.xlabel('x',fontsize=14)
            plt.ylabel('y',fontsize=14)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            plt.savefig(result_dir+"/Plotted_map/per_"+str(per)+"/"+str(trial)+"/"+name.decode()+"_plot.png",dpi=300)
            plt.close()

            fig, ax = plt.subplots()
            plot = ax.contourf(X,Y,Probability_log,np.arange(-15.00, -12.99, 0.01),cmap="jet",alpha=0.7, extend='both', antialiased=True)
            cbar = fig.colorbar(plot)
            tick_locator = ticker.MaxNLocator(nbins=11)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.set_label(r"$ \log {p(x_t|n_t)}$")
            img = Image.open("map/3ldk_9_local.PNG")
            ax.imshow(img,extent=[x_min,x_max,y_min,y_max])
            plt.xlabel('x',fontsize=12)
            plt.ylabel('y',fontsize=12)
            plt.setp(ax.get_xticklabels(), fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=12)
            plt.savefig(result_dir+"/Heat_map/per_"+str(per)+"/"+str(trial)+"/"+name.decode()+"_heatmap.png",dpi=300)
            plt.close()
            #"""

    per += PER
