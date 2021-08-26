import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import math
from scipy import stats
from statistics import mean

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set1')

result = sys.argv[1]

def cilen(arr, alpha=0.95):
    if len(arr) <= 1:
        return 0
    m, e, df = np.mean(arr), stats.sem(arr), len(arr) - 1
    interval = stats.t.interval(alpha, df, loc=m, scale=e)
    cilen = np.max(interval) - np.mean(interval)

    if math.isnan(cilen):
        cilen = 0

    return cilen

#file_name = ["kappa_statistic_mixture_mixture","kappa_statistic_global_mixture","kappa_statistic_local_mixture","kappa_statistic_local_local"]
file_name = ["pos_cost_specific"]
#models = ["spcoa","env_num_2","env_num_8","env_num_32"]
models = ["env_num_16"]
#x = np.array([0, 5, 10, 15, 20, 25, 30])
pers = [0, 20, 40, 60, 80, 100]
#pers_label = [r'$\frac{0}{20}$',r'$\frac{4}{20}$',r'$\frac{8}{20}$',r'$\frac{12}{20}$',r'$\frac{16}{20}$',r'$\frac{20}{20}$']
pers_label = [r'$\frac{0}{40}$',r'$\frac{8}{40}$',r'$\frac{16}{40}$',r'$\frac{24}{40}$',r'$\frac{32}{40}$',r'$\frac{40}{40}$']
#y = []
#e = []

fig = plt.figure()
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)

for fn in file_name:

    #f = "gibbs_result/similar_result/"+model+"/"+fn+".txt"
    f = "gibbs_result/sigverse_result/env_num_16/"+fn+".txt"
    data_all = np.loadtxt(f,delimiter=" ")

    df = pd.DataFrame({})
    data = [[] for i in range(len(pers))]

    for j in range(len(data_all)):
        for k, per in enumerate(pers):
            if data_all[j][0] == per:
                data[k].append(data_all[j][1])
    
    for k in range(len(pers)):
        df[pers_label[k]] = data[k]
        #print(np.average(data[k]))

df_melt = pd.melt(df)
sns.boxplot(x='variable',y='value', data=df_melt, palette=sns.color_palette("OrRd", 7), ax=ax,
            showmeans=True,meanprops={"marker":"o",
            "markerfacecolor":"white", 
            "markeredgecolor":"black",
            "markersize":"5"})
plt.plot([], [], 'd', color="black",label='outlier')
plt.plot([], [], 'o', markerfacecolor='white', markeredgecolor="black",label='mean')
ax.set_xlabel('Name given rate in a place in a new environment', fontsize=14)
ax.set_ylabel('cost', fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.legend(bbox_to_anchor=(1, 1.16), loc='best', borderaxespad=0, fontsize=12)
plt.ylim(0,1000)
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=12)
#plt.show()
plt.savefig("gibbs_result/"+result+"/Position_prediction_result/adaption/pos_cost_specific_box.png",dpi=1000)
