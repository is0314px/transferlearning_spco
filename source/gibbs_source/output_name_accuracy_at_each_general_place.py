#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import math
from scipy import stats
from statistics import mean, stdev

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set1')

result = sys.argv[1]
dic = sys.argv[2]

def cilen(arr, alpha=0.95):
    if len(arr) <= 1:
        return 0
    m, e, df = np.mean(arr), stats.sem(arr), len(arr) - 1
    interval = stats.t.interval(alpha, df, loc=m, scale=e)
    cilen = np.max(interval) - np.mean(interval)

    if math.isnan(cilen):
        cilen = 0

    return cilen

#models = ["spcoa","env_num_2","env_num_8","env_num_32"]
models = ["spcoa","spcoa+MI","spcotransfer19_16","spcotransfer19+MI_16","spcotransfer20_0","spcotransfer20_1","spcotransfer20_2","spcotransfer20_4","spcotransfer20_8","spcotransfer20_16","spcotransfer20+MI_0","spcotransfer20+MI_1","spcotransfer20+MI_2","spcotransfer20+MI_4","spcotransfer20+MI_8","spcotransfer20+MI_16"]
x = ["SpCoA","SpCoA\n+MI","SpCo\nTransfer'19\n16 env","SpCo\nTransfer'19\n+MI 16 env","SpCo\nTransfer'20\n0 env","SpCo\nTransfer'20\n1 env","SpCo\nTransfer'20\n2 env","SpCo\nTransfer'20\n4 env","SpCo\nTransfer'20\n8 env","SpCo\nTransfer'20\n16 env","SpCo\nTransfer'20\n+MI 0 env","SpCo\nTransfer'20\n+MI 1 env","SpCo\nTransfer'20\n+MI2 env","SpCo\nTransfer'20\n+MI4 env","SpCo\nTransfer'20\n+MI 8 env","SpCo\nTransfer'20\n+MI16 env"]

name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic, delimiter="\n", dtype='S' )
Names = []

file = 'gibbs_result/sigverse_result/Name_prediction_result/generalization/name_acc_each_general_place.txt'
for i, nd in enumerate(name_dic):
    if i > 6:
        Names.append(nd)

max_width = max(len(s) for s in name_dic)
f = open(file,'a')

for m, model in enumerate(models):
    try:
        print(x[m])
        f = open(file,'a')
        f.write(x[m]+"\n")
        f.close()

        for name in Names:
            #f = "gibbs_result/similar_result/"+model+"/"+fn+".txt"
            fn = "name_acc_"+name
            f = "gibbs_result/sigverse_result/"+model+"/Name_evaluation/"+fn+".txt"
            data_all = np.loadtxt(f,delimiter=" ")

            Mean = mean(data_all)
            Error = stdev(data_all)
            #Error = cilen(data_all)
            print(name+" "*(max_width-len(name)+1)+":"+repr(np.round(Mean,3))+"+-"+repr(np.round(Error,3)))
            f = open(file,'a')
            f.write(name+" "*(max_width-len(name)+1)+":"+repr(np.round(Mean,3))+"+-"+repr(np.round(Error,3))+"\n")
            f.close()

    except:
        pass
    
f.close()