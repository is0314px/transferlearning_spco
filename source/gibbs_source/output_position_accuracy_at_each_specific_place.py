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
models = ["spcotransfer19_16","spcotransfer19+MI_16","spcotransfer20_16","spcotransfer20+MI_16"]
x = np.array([0, 20, 40, 60, 80, 100])
#x = [r'$\frac{0}{20}$',r'$\frac{4}{20}$',r'$\frac{8}{20}$',r'$\frac{12}{20}$',r'$\frac{16}{20}$',r'$\frac{20}{20}$']
#x = [r'$\frac{0}{40}$',r'$\frac{8}{40}$',r'$\frac{16}{40}$',r'$\frac{24}{40}$',r'$\frac{32}{40}$',r'$\frac{40}{40}$']
dic_dim = 18
cl = np.array([1.0/dic_dim for i in range(len(x))])
print(cl)

name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic, delimiter="\n", dtype='S' )
Names = []

for i, nd in enumerate(name_dic):
    if i > 14:
        Names.append(nd)

max_width = max(len(s) for s in name_dic)

for md in models:
    y = []
    e = []
    for name in Names:

        #f = "gibbs_result/similar_result/"+model+"/"+fn+".txt"
        fn = "pos_acc_"+name
        f = "gibbs_result/sigverse_result/"+md+"/Position_evaluation/"+fn+".txt"
        data_all = np.loadtxt(f,delimiter=" ")

        data_per_0 = []
        data_per_20 = []
        data_per_40 = []
        data_per_60 = []
        data_per_80 = []
        data_per_100 = []

        for j in range(len(data_all)):
            if data_all[j][0] == 0:
                data_per_0.append(data_all[j][1])
            elif data_all[j][0] == 20:
                data_per_20.append(data_all[j][1])
            elif data_all[j][0] == 40:
                data_per_40.append(data_all[j][1])
            elif data_all[j][0] == 60:
                data_per_60.append(data_all[j][1])
            elif data_all[j][0] == 80:
                data_per_80.append(data_all[j][1])
            else:
                data_per_100.append(data_all[j][1])         

        means = np.array([mean(data_per_0),mean(data_per_20),mean(data_per_40),mean(data_per_60),mean(data_per_80),mean(data_per_100)])
        errors = np.array([stdev(data_per_0),stdev(data_per_20),stdev(data_per_40),stdev(data_per_60),stdev(data_per_80),stdev(data_per_100)])
        #errors = np.array([cilen(data_per_0),cilen(data_per_20),cilen(data_per_40),cilen(data_per_60),cilen(data_per_80),cilen(data_per_100)])]

        y.append(means)
        e.append(errors)

    y = np.array(y)
    e = np.array(e)

    file ='gibbs_result/sigverse_result/Position_prediction_result/adaption/position_acc_each_specific_place.txt'

    for p, per in enumerate(x):
        print(md+" name teaching rate "+repr(per)+"%")
        f = open(file,'a')
        f.write(md+" name teaching rate "+repr(per)+"%\n")
        f.close()

        for n, name in enumerate(Names):
            print(name+" "*(max_width-len(name)+1)+":"+repr(np.round(y[n,p],3))+"+-"+repr(np.round(e[n,p],3)))
            f = open(file,'a')
            f.write(name+" "*(max_width-len(name)+1)+":"+repr(np.round(y[n,p],3))+"+-"+repr(np.round(e[n,p],3))+"\n")
            f.close()

