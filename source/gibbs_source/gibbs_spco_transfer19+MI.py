#!/usr/bin/env python
# -*- coding:utf-8 -*-

##########################################
#Gibbs sampling for training spatial concept 
#Author Satoshi Ishibushi
#Editor Kazuya Asada (2017/09/29)
#Re-editor Keishiro Taguchi (2021/02/22)
#-==========================================-
##########################################
import unicodedata
import argparse
import numpy as np
import random
import string
import sys
import glob
import re
import math
import os
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
import time
import matplotlib.pylab as plt
sys.path.append("lib/")
import BoF
import Prob_Cal
import Multi
import file_read as f_r
import nonpara_tool
import show_ellipse
import shutil
import scipy.stats as ss
from PIL import Image
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    from sklearn.preprocessing import normalize
    import sklearn.cluster

start_time = time.time()
parser = argparse.ArgumentParser()

parser.add_argument(
    "Training_dataset_list",
    help = "Input training directory."
)

parser.add_argument(
    "Output_directory",
    help = "Output directory"
)

parser.add_argument(
    "dictionary",
    help = "Input dictionary"
)

parser.add_argument(
    "experiment_type",
    help = "Input experiment_type"
)

parser.add_argument(
    "--slide",
    default = 1,
    help = "Sliding num for reading training data."
)

parser.add_argument(
    "--vision",
    type = str,
    default = None,
    help = "vision_dirc"
)

parser.add_argument(
    "--iteration",
    type = int,
    default = 1000,
    help = "iteration"
)

parser.add_argument(
    "--name_not",
    type = int,
    default = None,
    help = "If you don't use name for environmet."
)

parser.add_argument(
    "--test_num",
    type = int,
    default = 1,
    help = "If you don't use function description for environmet."
)

args = parser.parse_args()
Training_list = np.loadtxt(args.Training_dataset_list , delimiter="\n", dtype="S") #directories of training dataset and test dataset
dic_txt = args.dictionary
ex_type = args.experiment_type

try:
    data_set_num = len(Training_list)
except TypeError:
    Training_list = np.array([Training_list])
    data_set_num = 1
Slide = int(args.slide)

vision_data_dir = "/vision_fc7_normalized/"  #directory where vision data exists
position_data_dir = "/position/"
hyper_para = np.loadtxt("hyper_parameter/hyper_parameter_spcotransfer.txt",delimiter = " #") #read hyper parameters
name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic_txt, delimiter = "\n", dtype = "S" ) #dictionary of name

PER = 20 #every PER percent
Stick_large_L = 15 #number of concept upper limit in stick breaking process
Stick_large_R = 20 #number of position distribution upper limit in stick breaking process
sigma_init = 100.0 #initial covariance value
iteration = 200 #number of iterarion
test_num = args.test_num #index of test dataset

if ex_type == "general":
    MAX_PER = 0
elif ex_type == "specific":
    MAX_PER = 100
else:
    sys.exit()

#========== Read hyper parameters ==========
alpha_v = hyper_para[0] #3
alpha_n = hyper_para[1] #0.01
kappa_0 = hyper_para[2] #0.05
nu_0 = hyper_para[3] #10
gamma = hyper_para[4] #10
gamma_0 = hyper_para[5] #0.2
beta = hyper_para[6] #3.0
vision_increment = hyper_para[7] #1.0
name_increment = hyper_para[8] #5000
delta_v = hyper_para[9] #300000
delta_n = hyper_para[10] #10000 
#delta_n = 500000 #被験者データセットを使う時
psi_0_x = hyper_para[11] #10.0
psi_0_y = hyper_para[12] #10.0
psi_0_sin = hyper_para[13] #0.5
psi_0_cos = hyper_para[14] #0.5
psi_0 = np.matrix([[psi_0_x,0.0,0.0,0.0],
                    [0.0,psi_0_y,0.0,0.0],
                    [0.0,0.0,psi_0_sin,0.0],
                    [0.0,0.0,0.0,psi_0_cos]])
epsilon = hyper_para[15] #0.1
#epsilon = 0.01 #被験者データセットを使う時


def position_data_read(directory,DATA_NUM,test_num):

    all_position = [] 

    for i in range(DATA_NUM):
        if  (i in test_num) == False:
            f = directory + repr(i) + ".txt"
            position = [] #(x,y,sin,cos)

            for line in open(f, 'r').readlines():

                try: #for similar
                    data=line[:-1].split('\t')
                    position +=[float(data[0].replace('\xef\xbb\xbf', ''))]

                except: #for sigverse
                    data=line[:-1].split(' ')
                    position +=[float(data[0].replace('\r', ''))]

                position +=[float(data[1].replace('\r', ''))]

            position = [position[0]*100,position[1]*100,position[2],position[3]] #x and y multiplied by 100
            all_position.append(position)

    return np.array(all_position)


def vision_data_read(directory,vision_increment,DATA_NUM,test_num):

    all_vision = []

    for i in range(DATA_NUM):

        if  (i in test_num) == False:
            f = directory + repr(i) + ".csv"
            vision = np.loadtxt(f,delimiter=",")

            try:
                vision = np.array(vision[:,0])

            except IndexError:
                pass
            
            vision = vision * vision_increment
            all_vision.append(vision)
    
    return np.array(all_vision)


def Name_data_read(directory,name_increment,DATA_NUM,test_num, e):

    name_data_set = []

    for i in range(DATA_NUM):
        name_data = [0 for w in range(len(name_dic))]

        if  (i in test_num) == False:
            try:
                if e < data_set_num - 1:
                    file = directory + "sentence" + repr(i) + ".txt"
                else:
                    file = directory + "sentence" + repr(i) + ".txt"

                data = np.loadtxt(file, delimiter=" ", dtype='S')
                data_n = data.tolist()

                if isinstance(data_n, list) == False:
                    for w, dictionry in enumerate(name_dic):
                        if data_n == dictionry:
                            name_data[w] += name_increment

                else:
                    for d in data_n:
                        for w, dictionry in enumerate(name_dic):
                            if d == dictionry:
                                name_data[w] += name_increment

            except IOError:
                pass

            name_data = np.array(name_data)
            name_data_set.append(name_data)

    return np.array(name_data_set)


def test_data_read(file,test_num):

    test_data_num = []

    for line in open(file, 'r').readlines():

        num = line[:-1].split(',') #delete "\n"
        
        for n in num:
            try:
                test_data_num.append(int(n))

            except ValueError:
                pass
        
    return test_data_num


def init_mu(posdist_num,position,DATA_NUM):
    mu = []
    kmeans = sklearn.cluster.KMeans(n_clusters = posdist_num, random_state = random.randint(1,100)).fit(position)

    for j in range(posdist_num):
        index = int(random.uniform(0,DATA_NUM))
        p = kmeans.cluster_centers_[j]
        mu.append(p)

    return np.array(mu)


def gibbs_sampling(per):

    vision_set = []
    position_set = []
    name_set = []
    C_t_set = []
    R_t_set = []
    G_set = []
    pi_set = []
    Sigma_set = []
    mu_set = []
    data_num = []
    concept_num = Stick_large_L
    posdist_num = Stick_large_R
    MAP_X = [0 for e in range(data_set_num)] #MAX x value of map
    MAP_Y = [0 for e in range(data_set_num)] #MAX y value of map
    map_x = [0 for e in range(data_set_num)] #min x value of map
    map_y = [0 for e in range(data_set_num)] #min y value of map
    mu_0_set = []
    print("")

    G_0 = nonpara_tool.stick_breaking(gamma_0,concept_num) #initialize G_0

    for e, dir in enumerate(Training_list):

        #=============== Get mu_0 ===============
        
        env_para = np.genfromtxt(dir + "/Environment_parameter.txt",dtype = None,delimiter = " ") #read environment parameter

        MAP_X[e] = float(env_para[0][1])
        MAP_Y[e] = float(env_para[1][1])
        map_x[e] = float(env_para[2][1])
        map_y[e] = float(env_para[3][1])
        DATA_initial_index = int(env_para[5][1])
        DATA_last_index = int(env_para[6][1])
        DATA_NUM = DATA_last_index - DATA_initial_index + 1 #calcurate data number

        map_center_x = ((MAP_X[e] - map_x[e])/2) + map_x[e]
        map_center_y = ((MAP_Y[e] - map_x[e])/2) + map_y[e]
        mu_0 = np.array([map_center_x,map_center_y,0,0]) #get mu_0 in every environment
        mu_0_set.append(mu_0)

        #=============== Read data ===============

        if e < len(Training_list) - 1: #directory where training name data exists
            name_data_dir = "/sentence/per_100/"
            test_data_list = dir + "/test_num.txt"
            test_data_num = test_data_read(test_data_list,test_num)

        else: #directory where test name data exists
            name_data_dir = "/sentence/per_" + repr(per) + "/"
            test_data_list = dir + "/test_num.txt"
            test_data_num = test_data_read(test_data_list,test_num)

        position_dir = dir + position_data_dir
        vision_dir = dir + vision_data_dir #if args.vision == None:
        name_dir = dir + name_data_dir

        position = position_data_read(position_dir,DATA_NUM,test_data_num) #read positon data
        position_set.append(position)

        vision = vision_data_read(vision_dir,vision_increment,DATA_NUM,test_data_num) #read vision data
        vision_set.append(vision)

        name = Name_data_read(name_dir,name_increment,DATA_NUM,test_data_num,e) #read name data
        name_set.append(name)

        #=============== Print infomation of read data ================

        print("Read environment directory: " + dir)
        print("Name vector sum")
        print(sum(name/name_increment))
        print("")
        
        #if test environment, next processing is done
        if len(test_data_num) > 1:
            DATA_NUM = DATA_NUM - len(test_data_num)

        Learnig_data_num = (DATA_last_index - DATA_initial_index + 1) / Slide #learning data number
        data_num.append(DATA_NUM)

        #=============== Initialize parameters ===============

        G = np.random.dirichlet(G_0 + gamma) #initialize G
        G_set.append(G)

        pi_e = []
        for i in range(concept_num):
            pi = nonpara_tool.stick_breaking(beta,Stick_large_R) #initialize pi
            pi_e.append(pi)
        pi_set.append(pi_e)

        c_t = [1000 for n in xrange(DATA_NUM)] #initialize c_t
        C_t_set.append(c_t)

        R_t = [1000 for n in xrange(DATA_NUM)] #initialize R_t
        R_t_set.append(R_t)

        mu = init_mu(posdist_num,position,DATA_NUM) #initialize mu
        mu_set.append(mu)
        
        Sigma = np.array([[[sigma_init,0.0,0.0,0.0],
                           [0.0,sigma_init,0.0,0.0],
                           [0.0,0.0,sigma_init,0.0],
                           [0.0,0.0,0.0,sigma_init]]   for i in range(posdist_num)]) #initialize Sigma
        Sigma_set.append(Sigma)

    VISION_DIM = len(vision_set[0][0])
    NAME_DIM = len(name_set[0][0])
    
    phi_v = np.array([[float(1.0)/VISION_DIM for i in range(VISION_DIM)] for j in range(concept_num)]) #initialize phi^v
    phi_n = np.array([[float(1.0)/NAME_DIM for i in range(NAME_DIM)] for j in range(concept_num)]) #initialize phi^n

    posdist_choice = [dr for dr in range(posdist_num)] #[0,1,2,...,posdist - 1]
    concept_choice = [dc for dc in range(concept_num)] #[0,1,2,...,concept - 1]

    for iter in xrange(iteration): #iteration == 200

        print("iteration "+ repr(iter + 1))
        posdist_count = [[0.0 for i in range(posdist_num)] for j in range(data_set_num)] #store the number of position distributions in all environments
        concept_count_set = [] #store the number of concepts in all environments
        #Total_name_e_set = []

        for e in xrange(data_set_num):

            #=============== Sample R_t ===============
            concept_posdist_count = [[0.0 for i in range(posdist_num)] for j in range(concept_num)] #store the number of position distributions in concepts
            concept_count = [0.0 for i in range(concept_num)] #store the number of concepts in environment e
            gauss_prob_set = np.zeros((posdist_num,data_num[e]),dtype = float) #store log(p(x_t|mu_{R_t},Sigma_{R_t}) + p(R_t|pi_{C_t}))
            
            if iter == 0:
                C_t = np.random.randint(concept_num,size = data_num[e]) #initialize C_t
            else:
                C_t = C_t_set[e]

            pi_e = np.array(pi_set[e])
            pi_data = np.array([pi_e[C_t[d]] for d in range(data_num[e])])
            pi_data = np.log(pi_data)

            for i in range(posdist_num):
                gauss_prob = ss.multivariate_normal.logpdf(position_set[e],mu_set[e][i],Sigma_set[e][i]) + pi_data[:,i] #caluculate log(p(x_t|mu_{R_t},Sigma_{R_t})+p(R_t|pi_{C_t}))
                gauss_prob_set[i] += gauss_prob 

            gauss_prob_set = gauss_prob_set.T
            max_posdist = np.max(gauss_prob_set,axis = 1)
            gauss_prob_set = gauss_prob_set - max_posdist[:,None]
            gauss_prob_set = np.exp(gauss_prob_set)
            sum_set = np.sum(gauss_prob_set,axis = 1)
            gauss_prob_set = gauss_prob_set / sum_set[:,None]

            for d in xrange(0,data_num[e],Slide):
                R_t_set[e][d] = np.random.choice(posdist_choice,p = gauss_prob_set[d]) #sample R_t
                posdist_count[e][R_t_set[e][d]] += 1.0
            
            R_t = R_t_set[e]

            #=============== Sample C_t ===============
            multi_prob_set = np.zeros((concept_num,data_num[e]),dtype = float) #store log(p(v_t|phi^v_{C_t}) + p(n_t|phi^n_{C_t}) + p(C_t|G))

            phi_v_log = np.log(phi_v)
            phi_n_log = np.log(phi_n)
            pi_data = np.array([pi_e.T[R_t[d]] for d in range(data_num[e])])
            pi_data = np.log(pi_data)
            G_log = np.log(G_set[e])
            
            for i in range(concept_num):
                vision_prob = vision_set[e].dot(phi_v_log[i])
                name_prob = name_set[e].dot(phi_n_log[i])
                modal_prob = vision_prob + name_prob + pi_data[:,i]
                modal_prob = modal_prob + G_log[i] 
                multi_prob_set[i] += modal_prob

            multi_prob_set = multi_prob_set.T
            max_concept = np.max(multi_prob_set,axis = 1)
            multi_prob_set = multi_prob_set - max_concept[:,None]
            multi_prob_set = np.exp(multi_prob_set)
            sum_concept_set= np.sum(multi_prob_set,axis = 1)
            multi_prob_set = multi_prob_set / sum_concept_set[:,None]

            for d in xrange(0,data_num[e],Slide):
                C_t_set[e][d] = np.random.choice(concept_choice,p = multi_prob_set[d]) #sample C_t
                concept_count[C_t_set[e][d]] += 1.0
                concept_posdist_count[C_t_set[e][d]][R_t[d]] += 1.0

            #=============== Sample mu and Sigma ========== 
            for r in xrange(posdist_num):
                pos_r = [] #store position data in position distribution r
                
                #===== Calculate average =====
                for d in xrange(data_num[e]):
                    if R_t_set[e][d] == r:
                        pos_r.append(position_set[e][d])
                        
                sum_pose = np.zeros(len(mu[0])) #([0.0,0.0,0.0,0.0])
                for i in xrange(len(pos_r)):
                    for j in xrange(len(mu[0])):
                        sum_pose[j] += pos_r[i][j]

                bar_pose = np.zeros(len(mu[0])) #([0.0,0.0,0.0,0.0])
                for i in xrange(len(mu[0])):
                    if sum_pose[i] != 0:		 	
                        bar_pose[i] = sum_pose[i] / len(pos_r)

			    #===== Calculate Mu =====
                Mu = (kappa_0 * mu_0_set[e] + len(pos_r) * bar_pose) / (kappa_0 + len(pos_r)) #Mu updated

                #===== Calculate Psi =====
                bar_pose_matrix = np.matrix(bar_pose)
                Matrix_R = np.zeros([len(mu[0]),len(mu[0])])

                for i in xrange(len(pos_r)):
                    pos_r_matrix = np.matrix(pos_r[i])
                    Matrix_R += ((pos_r_matrix - bar_pose_matrix).T * (pos_r_matrix - bar_pose_matrix))

                ans = ((bar_pose_matrix - mu_0_set[e]).T * (bar_pose_matrix - mu_0_set[e])) * ((kappa_0 * len(pos_r)) / (kappa_0 + len(pos_r)))
                Psi = psi_0 + Matrix_R + ans #Psi updated
    		 	
    		 	#===== Caluculate Kappa,Nu =====
                Kappa = kappa_0 + len(pos_r) #Kappa updated
                Nu = nu_0 + len(pos_r) #Nu updated

    		 	#===== Sample mu and Sigma from wishrt distribution =====
                Sigma_set[e][r] = Prob_Cal.sampling_invwishartrand(Nu,Psi) #sample Sigma
                Sigma_temp = Sigma_set[e][r] / Kappa
                mu_set[e][r] = np.random.multivariate_normal(Mu,Sigma_temp) #sample Mu

                if len(pos_r) == 0: #if no asigned data
                    p = np.array([random.uniform(map_x[e],MAP_X[e]),random.uniform(map_y[e],MAP_Y[e]),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)])
                    mu_set[e][r] = p
                    Sigma_set[e][r] = np.array([[sigma_init,0.0,0.0,0.0],
                                                [0.0,sigma_init,0.0,0.0],
                                                [0.0,0.0,sigma_init,0.0],
                                                [0.0,0.0,0.0,sigma_init]])

            #=============== Sample pi and G =============== 
            concept_count = [0.0 for i in range(concept_num)] 

            for c in xrange(concept_num): 
                vision_e_c = []
                name_e_c = []

                for d in xrange(data_num[e]):
                    if C_t_set[e][d] == c:
                        concept_count[c] += 1.0
                
                pi_set[e][c] = np.random.dirichlet(concept_posdist_count[c] + beta) #sample pi

            G_set[e] = np.random.dirichlet(concept_count + (gamma * G_0) + 50) + 1e-100 #sample G
            #G_set[e] = np.random.dirichlet(concept_count + (gamma * G_0)) + 1e-100 #sample G
            concept_count_set.append(concept_count)
	    
        #=============== Sample phi^v and phi^n and G_0 ===============
        total_vision_set = [] #store the number of vision (visual feature)
        total_name_set = [] #store the number of name

        for c in xrange(concept_num):

            vision_c = []
            name_c = []

            for e in xrange(data_set_num): #change point
                for d in xrange(data_num[e]):
                    if C_t_set[e][d] == c:
                        vision_c.append(vision_set[e][d])
                        name_c.append(name_set[e][d])

            total_vision = BoF.bag_of_feature(vision_c,VISION_DIM)
            total_vision = total_vision + alpha_v
            total_vision_set.append(total_vision)
            phi_v[c] = np.random.dirichlet(total_vision) + 1e-100 #sample phi_v
            
            total_name = BoF.bag_of_feature(name_c,NAME_DIM)
            total_name = total_name + alpha_n
            total_name_set.append(total_name)
            phi_n[c] = np.random.dirichlet(total_name) + 1e-100 #sample phi_n
            
            if len(vision_c) == 0:
                phi_v[c] = np.array([float(1.0) / VISION_DIM for i in range(VISION_DIM)])
                phi_n[c] = np.array([float(1.0) / NAME_DIM for i in range(NAME_DIM)])

        total_concept_count = np.sum(np.array(concept_count_set),axis=0)
        total_concept_count = total_concept_count + gamma_0
        G_0 = np.random.dirichlet(total_concept_count) #sample G_0

        #=============== Reconstruct word distribution by mutual information ===============     
        mutual_info = [[0.0 for i in range(NAME_DIM)] for j in range(concept_num)] #mutual information
        prob_n = [0.0 for i in range(NAME_DIM)]

        for n in range(NAME_DIM):
            for c in range(concept_num):   
                prob_n[n] += phi_n[c][n] * G_0[c]

        for c in range(concept_num):
            for n in range(NAME_DIM):
                t_t_prob = (phi_n[c][n] * G_0[c]) / (prob_n[n] * G_0[c])
                t_t_prob = (phi_n[c][n] * G_0[c]) * math.log(t_t_prob) #P(n_t,C_t|Theta) * log()
                
                t_f_prob = ((1.0-phi_n[c][n]) * G_0[c]) / ((1.0 - prob_n[n]) * G_0[c])
                if t_f_prob == 0.0:
                    t_f_prob = 1e-100
                t_f_prob = ((1.0-phi_n[c][n]) * G_0[c]) * math.log(t_f_prob) #P(n_t_ber,C_t|Thera) * log()
                
                f_t_prob = ((1.0-G_0[c]) * phi_n[c][n]) / (prob_n[n] * (1.0-G_0[c]))
                if f_t_prob == 0.0:
                    f_t_prob = 1e-100
                f_t_prob = ((1.0-G_0[c]) * phi_n[c][n]) * math.log(f_t_prob) #P(n_t,C_t_ber|Theta) * log()
                
                f_f_prob = ((1.0-G_0[c]) * (1.0-phi_n[c][n])) / ((1.0-prob_n[n]) * (1.0-G_0[c]))
                if f_f_prob == 0.0:
                    f_f_prob = 1e-100
                f_f_prob = ((1.0-G_0[c]) * (1.0-phi_n[c][n])) * math.log(f_f_prob) #P(n_t_ber,C_t_ber|Theta) * log()
            
                mutual_info[c][n] = t_t_prob + t_f_prob + f_t_prob + f_f_prob

        max_of_mutual_info = np.max(mutual_info, axis = 0)
        min_of_mutual_info = np.min(mutual_info, axis = 0) 
        max_width = max(len(s) for s in name_dic)

        #You can confilm max of mutual information of each word
        """
        print("Max of mutual infomation")
        for n in range(NAME_DIM):
            print(name_dic[n]+" "*(max_width-len(name_dic[n]))+" :"+repr(np.round(max_of_mutual_info[n],3)))
        """

        for n in range(NAME_DIM): #reconstruct
            if max_of_mutual_info[n] < epsilon:
                for c in range(concept_num): 
                    phi_n[c][n] = 1e-100

        for c in range(concept_num): #normalize
            Sum = np.sum(phi_n[c])
            for n in range(NAME_DIM):
                phi_n[c][n] = phi_n[c][n] / Sum

        #=========================================================

        if ((iter + 1) % iteration) == 0:
            
            exist_concept_num = 0 
            for i in range(concept_num):
                if concept_count[i] > gamma_0:
                    exist_concept_num += 1

            exist_posdist_num = [0 for e in range(data_set_num)]
            for e in range(data_set_num):
                for r in range(posdist_num):
                    if posdist_count[e][r] > 0:
                        exist_posdist_num[e] += 1
            
            #========== Save ==========
            new_env_idx = len(Training_list)-1
            Out_put_dir = args.Output_directory + "/per_"+repr(per)+"_iter_"+repr(iter+1)

            try: 
                os.mkdir(Out_put_dir)
            except OSError:
                shutil.rmtree(Out_put_dir)
                os.mkdir(Out_put_dir)

            #save finish time
            finish_time = time.time() - start_time
            f = open(Out_put_dir + "/time.txt","w")
            f.write("time:" + repr(finish_time) +" seconds.")
            f.close()

            f = open(Out_put_dir + "/training_dataset",'w')
            for i,d in enumerate(Training_list):
                if i == len(Training_list)-1:
                    w = "New env "+repr(i) +":  "+d +"\n"
                else:
                    w = "Known env "+repr(i) +":  "+d +"\n"
                f.write(w)

            f.close()
            
            #learning result. You can confilm result of clustering and word distribution (\phi_n)
            """
            color = ['r','b','g','maroon','c','m','springgreen','steelblue','silver','gold','navy','mediumpurple','yellowgreen','tan','tomato']

            fig, ax = plt.subplots()
            
            env_names = ['known','new']
            env_index = [0, data_set_num-1]

            for ei, env_name in enumerate(env_names):
                e = env_index[ei]
                fig, ax = plt.subplots()

                for i in range(data_num[e]):
                    plt.scatter(position_set[e][i,0],position_set[e][i,1],c = color[C_t_set[e][i]],marker='.')
                
                for r in range(Stick_large_R):
                    show_ellipse.main(mu_set[e][r],Sigma_set[e][r],Stick_large_R,ax)
                
                if env_name == 'new':
                    img = Image.open("map/3LDK_9.PNG")
                    coordinate = np.loadtxt("map/3LDK_9.txt", delimiter = " #", dtype=float) 
                    plt.imshow(img,extent=(coordinate[0],coordinate[1],coordinate[2],coordinate[3]))
                
                else:
                    img = Image.open("map/1LDK_8.PNG")
                    coordinate = np.loadtxt("map/1LDK_8.txt", delimiter = " #", dtype=float) 
                    plt.imshow(img,extent=(coordinate[0],coordinate[1],coordinate[2],coordinate[3]))

                ax.set_xlim(coordinate[0],coordinate[1])
                ax.set_ylim(coordinate[2],coordinate[3])
                plt.xlabel('x',fontsize=14)
                plt.ylabel('y',fontsize=14)
                plt.setp(ax.get_xticklabels(), fontsize=14)
                plt.setp(ax.get_yticklabels(), fontsize=14)

                plt.savefig("gibbs_result/sigverse_result/spcotransfer19+MI_"+repr(data_set_num-1)+"/Visualize_learning_result/spatial_concept_"+env_name+"/per_"+repr(x)+".png",dpi=300) #spatial_concept
                plt.close()

            for c in range(Stick_large_L):
                plt.bar(name_dic,phi_n[c],tick_label=name_dic,align="center",color=color[c])
                plt.xticks(rotation='vertical')
                plt.tight_layout()
                plt.ylim(0,1)
                plt.savefig("gibbs_result/sigverse_result/spcotransfer19+MI_"+repr(data_set_num-1)+"/Visualize_learning_result/spatial_concept_global/per_"+repr(per)+"/concept_"+repr(c)+".png") #spatial_concept
                plt.close()
            #"""

            #=============== Save local parameter ===============
            for i in range(data_set_num):
                os.mkdir(Out_put_dir+"/dataset"+repr(i))
                os.mkdir(Out_put_dir+"/dataset"+repr(i)+"/mu")
                os.mkdir(Out_put_dir+"/dataset"+repr(i)+"/sigma")
                #np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/data_concept.txt",C_t_set[i])
                #np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/data_posdist.txt",R_t_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/concept_count.txt",concept_count_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/posdist_count.txt",posdist_count[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/pi.csv",pi_set[i])
                np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/G.txt",G_set[i])              
                f = open(Out_put_dir+"/dataset"+repr(i)+"/Parameter.txt","w")
                f.write("max_x_value_of_map: "+repr(MAP_X[i])+
                    "\nMax_y_value_of_map: "+repr(MAP_Y[i])+
                    "\nMin_x_value_of_map: "+repr(map_x[i])+
                    "\nMin_y_value_of_map: "+repr(map_y[i])+
                    "\nConcept_num: "+repr(exist_concept_num)+
                    "\nPosition_distribution_num: "+repr(exist_posdist_num[i])+
                    "\nData_num: "+repr(data_num[i])+
                    "\nDataset: "+Training_list[i]+
                    "\nName_dim: "+repr(NAME_DIM)+
                    "\nVision_dim: "+repr(VISION_DIM)
                    )
                f.close()

                """
                f = open(Out_put_dir+"/dataset"+repr(i)+"/data_concept_posdist.txt","w")
                for d in range(data_num[i]):
                    if w > 0:
                        f.write("data:"+repr(d)+" C_t:"+repr(C_t_set[i][d])+" R_t:"+repr(R_t_set[i][d])+"\n")
                f.close()
                """

                #=============== Save position distrbution parameter ===============
                for j in xrange(posdist_num):
                    np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/mu/gauss_mu"+repr(j)+".csv",mu_set[i][j])
                    np.savetxt(Out_put_dir+"/dataset"+repr(i)+"/sigma/gauss_sigma"+repr(j)+".csv",Sigma_set[i][j])
                
            #=============== Save global parameter ===============
            #np.savetxt(Out_put_dir+"/dataset"+repr(data_set_num-1)+"/bag_of_vision_local.csv",total_vision_e_set)
            #np.savetxt(Out_put_dir+"/dataset"+repr(data_set_num-1)+"/bag_of_name_local.csv",total_name_e_set)
            #np.savetxt(Out_put_dir+"/bag_of_vision.csv",total_vision_set)
            #np.savetxt(Out_put_dir+"/bag_of_name.csv",total_name_set)
            np.savetxt(Out_put_dir+"/phi_v.csv",phi_v)
            np.savetxt(Out_put_dir+"/phi_n.csv",phi_n)
            #np.savetxt(Out_put_dir+"/mutual_info.csv",np.round(mutual_info, decimals=3))

            f = open(Out_put_dir+"/hyper_parameter.txt","w")
            f.write("alpha_v: "+repr(alpha_v)+("\nalpha_n: ")+repr(alpha_n)
                +("\ngamma_0: ")+repr(gamma_0)+"\ngamma: "+repr(gamma)
                +"\nbeta: "+repr(beta)+("\nkappa_0: ")+repr(kappa_0)+("\nnu_0: ")+repr(nu_0)
                +"\ndelta_v: "+repr(delta_v)+"\ndelta_n: "+repr(delta_n)+"\nepsilon: "+repr(epsilon)
                +"\nvision_increment: "+repr(vision_increment)+"\nname_increment: "+repr(name_increment)
                +"\nPsi_0: ["+repr(psi_0[0][0])+"\n"+repr(psi_0[1][0])+"\n"+repr(psi_0[2][0])+"\n"+repr(psi_0[3][0])+"]"
                )
            f.close()
            
    print(name_data_dir)

if __name__ == "__main__":

    per = 0 #name teaching rate
    while per <= MAX_PER: #MAX_PER
        gibbs_sampling(per)
        per = per + PER