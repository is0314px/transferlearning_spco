#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np 
import os
import sys
import math
from numpy.linalg import inv, cholesky
import scipy.stats as ss
import matplotlib.pylab as plt
import pandas as pd
import shutil
from PIL import Image

train_list = sys.argv[1] 
result_dir = sys.argv[2]
dic_txt = sys.argv[3]
ex_type = sys.argv[4]
#on_or_off = sys.argv[5]

name_list = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic_txt, delimiter="\n", dtype='S' )

def Multi_prob(data,phi): #log p(*|phi^*)
    phi += 1e-300
    phi_log = np.log(phi)
    prob = data.dot(phi_log.T)
    return prob

def normalize(probs): 
    prob_factor = 1.0 / sum(probs)
    return [prob_factor * p for p in probs]

def Name_data_read(file,name_increment): #read name data
    name_data = [0 for w in range(len(name_list))]    
    data = np.loadtxt(file, delimiter=' ', dtype='S' )
    data_l = data.tolist()
    
    if isinstance(data_l, list) == False:
        for w,dictionry in enumerate(name_list):
            if data_l == dictionry:
                name_data[w] += name_increment
    else:
        for d in data_l:
            for w,dictionry in enumerate(name_list):   
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

mute = 1 #if mutual infomation is off, mute is 0
test_num = 0 #python2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_similar/path_num${e}_D.txt gibbs_result/similar_result/env_num_$e transfer_similar/name_dic_local.txt
best = 1 #rapython2.7 predict_name.py ../gibbs_dataset/dataset_path_txt/transfer_similar/path_num${e}_D.txt gibbs_result/similar_result/env_num_$e transfer_similar/name_dic_local.txt
out_put = "name_estimate_value.txt" #Check file name!
name_out_put = "estimate_sample_list.txt"

per = 0
if ex_type == "global":
    max_per = 0
elif ex_type == "local":
    max_per = 100
else:
    print("ERROR")
    sys.exit()

try:
    data_set_num=len(Training_list) #number of trained rooms
except TypeError:
    Training_list=np.array([Training_list])
    data_set_num=1

while per <= max_per:

    parameter_dir = result_dir+"/per_"+repr(per)+"_iter_200/dataset"+repr(int(test_dataset_index))

    pi = np.loadtxt(parameter_dir+"/pi.csv") #read pi
    G = np.loadtxt(parameter_dir+"/G.txt") #read G
    mutual_info = np.loadtxt(parameter_dir+"/mutual_info.csv") #read mutual infomation I(n_t,C_t|Theta)

    region_num = len(pi[0])
    sigma_set = []
    mu_set = []
    region_count = np.loadtxt(parameter_dir+"/posdist_count.txt")

    for i in range(region_num):
        sigma = np.loadtxt(parameter_dir+"/sigma/gauss_sigma"+repr(i)+".csv") #read sigma
        mu = np.loadtxt(parameter_dir+"/mu/gauss_mu"+repr(i)+".csv") #read mu
        sigma_set.append(sigma)
        mu_set.append(mu)

    phi_n = np.loadtxt(parameter_dir+"/theta_n.csv") #read phi^n (for transfer learning model)
    try:
        phi_v = np.loadtxt(parameter_dir+"/theta_v.csv") #read phi^n (for transfer learning model)
    except:
        pass

    prob_n = normalize(np.sum(phi_n,axis=0))[0] 
    f_result = open(parameter_dir+"/"+out_put,'w')
    train_dir = Training_list[int(test_dataset_index)]

    if data_set_num < len(Training_list): 
        test_file = train_dir + "/test_num_all.txt"
    else: #read test number
        test_file = train_dir + "/test_num.txt"

    test_data_num = test_data_read(test_file,test_num)
    sample_test_data_num = [203,210,225,230,246,254,266,277,288,294]
    #sample_test_data_num = [210,225,294]

    test_vision_set = []
    test_pose_set = []
    test_name_set = []
    point_test_set = []

    for i in sample_test_data_num:
        try:
            test_vision = np.loadtxt(train_dir+"/vision_fc7_normalized/"+repr(i)+".csv",delimiter=",") #read vision data
            test_vision_set.append(test_vision)
        except:
            pass

        f = train_dir+"/position/"+repr(i)+".txt"
        test_pose = []

        for pos in open(f, 'r').readlines(): #read position data
            """
            try: #sigverse
                data = pos[:-1].split('\t')
                xy = data[0].replace('\xef\xbb\xbf', '')
            except: #realworld
                data = pos[:-1].split(' ')
                xy = data[0].replace('\r', '')
            """
            
            data = pos[:-1].split('\t')
            try:
                data = pos[:-1].split('\t')
                test_pose +=[float(data[0].replace('\xef\xbb\xbf', ''))]
            except:
                data = pos[:-1].split(' ')
                test_pose +=[float(data[0].replace('\r', ''))]

            test_pose +=[float(data[1].replace('\r', ''))]

            #sc = data[1].replace('\r', '')
            #test_pose.append(xy)
            #test_pose.append(sc)
        test_pose = [test_pose[0]*100,test_pose[1]*100,test_pose[2],test_pose[3]] #x and y multiplied by 100 注意！！
        test_pose_set.append(test_pose)
        
        try:
            if (e+1) < len(Training_list):
                test_name = Name_data_read(train_dir+"/name/"+repr(per_num)+"/word"+repr(i)+".txt",1) # Check path!
            else:
                test_name = Name_data_read(train_dir+"/name/"+repr(per_num)+"/word"+repr(i)+".txt",1) # Check path!
            point_temp = [0.0 for k in range(len(name_list))]
            point_test_name = [0.0 for k in range(len(name_list))]
            index = np.argsort(point_temp)[::-1]

            for rank in range(2):
                point_test_name[index[rank]] = point_temp[index[rank]]

            point_test_set.append(point_test_name)
            test_name_set.append(test_name)

        except:
            test_name = [0 for w in range(len(name_list))]
            test_name = np.array(test_name)

            point_temp = [0.0 for k in range(len(name_list))]  
            point_test_name = [0.0 for k in range(len(name_list))]
            index = np.argsort(point_temp)[::-1]
            for rank in range(2):
                point_test_name[index[rank]] = point_temp[index[rank]]
            point_test_set.append(point_test_name)
            test_name_set.append(test_name)

    test_vision_set = np.array(test_vision_set)
    test_pose_set = np.array(test_pose_set)
    test_name_set = np.array(test_name_set)

    #========== estimate ==========
    gauss_prob_set = np.zeros((region_num,len(sample_test_data_num)),dtype=float)
    for r in range(region_num):
        gauss_prob = ss.multivariate_normal.logpdf(test_pose_set,mu_set[r],sigma_set[r])    
        gauss_prob_set[r] += gauss_prob

    gauss_prob_set = gauss_prob_set.T
    max_region = np.max(gauss_prob_set,axis=1)
    gauss_prob_set = gauss_prob_set -max_region[:,None]
    gauss_prob_set = np.exp(gauss_prob_set)
    sum_set = np.sum(gauss_prob_set,axis=1)
    gauss_prob_set = gauss_prob_set / sum_set[:,None]


    for i ,idx in enumerate(sample_test_data_num):
        class_prob = np.array([0.0 for k in range(len(pi))])
        for c in range(len(pi)):
            try:
                class_prob[c] += Multi_prob(test_vision_set[i],phi_v[c]) # * p(v_t|phi^v_c)
                class_prob[c] += math.log(G[c]) # * p(c|G)
            except:
                class_prob[c] += math.log(G[c])

            region_prob=0.0
            for r in range(region_num):
                if region_count[r]>0:
                    region_prob += pi[c][r]*gauss_prob_set[i][r] # * p(r|pi_c) * p(x_t|mu_r,sigma_r)

            if region_prob != 0.0:
                class_prob[c] += math.log(region_prob)

        max_c = np.argmax(class_prob)    
        class_prob -= class_prob[max_c]
        class_prob = np.exp(class_prob)    
        class_prob = normalize(class_prob)

        prob1_1 = np.array([0.0 for k in range(len(name_list))]) #initialize p(n_t|v_t.x_t)

        for n in range(len(name_list)):
            for c in range(len(pi)):
                if mute == 0:
                    prob1_1[n] += class_prob[c]*phi_n[c][n] # * p(n_t|phi^n_c)
                else:
                    prob1_1[n] += class_prob[c]*phi_n[c][n]*mutual_info[c][n] # * p(n_t|phi^n_c) * I(n_t,C_t) #default
        
        prob1_1 = normalize(prob1_1)
        name_prob = prob1_1
        index = np.argsort(prob1_1)[::-1]
        prob1_1 = np.sort(prob1_1)[::-1]
        #f_result.write("Data: "+repr(test_data_num[i])+"\n") #original

        colors = ["lavender","lavender"]
        names = []
        probs = []

        for j in range(3): #3 best
            #for n in name_list:
                #if name_list[index[j]] == n:
                    #name = n
            names.append(name_list[index[j]])
            probs.append(str('{:.4f}'.format(prob1_1[j])))

        df = pd.DataFrame({
        'Word':names,
        'Probability':probs})[['Word','Probability']]

        plt.rcParams["font.size"] = 12
        fig, ax = plt.subplots(figsize=(4,2))
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText = df.values, colLabels = df.columns,colColours=colors,loc = 'center', bbox = [0,0,1,1])
        ax.set_title('3 best of predicted words', fontsize = 12)
        plt.savefig('gibbs_result/sigverse_result/Name_prediction_result/3best_table/env_num_16/per_'+repr(per)+'/'+repr(idx)+'.png', dpi = 1000)
                    
        # Edited 2017/11/20
        f_result.write("\n")
    f_result.close()

    per = per + 20

fig, ax = plt.subplots()
ax.invert_xaxis()
img = Image.open("map/3ldk_9_local.PNG") 
plt.imshow(img)
x_min = -590
x_max =  590
y_min = -470
y_max =  470
plt.imshow(img,extent=(x_min,x_max,y_min,y_max))
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
#ax.tick_params(labelbottom="false",bottom="false")
#ax.tick_params(labelleft="false",left="false")
#ax.set_xticklabels([]) 
#box("false")

for i ,idx in enumerate(sample_test_data_num):
    x = float(test_pose_set[sample_test_data_num.index(idx)][0])
    y = float(test_pose_set[sample_test_data_num.index(idx)][1])
    #dy = float(test_pose_set[test_data_num.index(idx)][2])*50
    #dx = float(test_pose_set[test_data_num.index(idx)][3])*50

    plt.scatter(x,y,c = 'black',marker = 'o',s = 50,alpha=0.7)
    #ax.annotate(idx, (x, y))
    #plt.quiver(x,y,dx,dy,angles='xy',scale_units='xy',scale=1,color='mediumblue')

    shutil.copyfile("../gibbs_dataset/sigverse/3LDK_9/image/"+repr(idx)+".png", "gibbs_result/sigverse_result/Name_prediction_result/3best_table/Images/data"+repr(idx)+".png")

plt.savefig("gibbs_result/sigverse_result/Name_prediction_result/3best_table/Positions/positions_on_map.png",dpi=1000)
