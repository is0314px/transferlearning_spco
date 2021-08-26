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
x = ["SpCoA","SpCoA\n+MI","SpCo\nTransfer'19\n16 env","SpCo\nTransfer'19\n+MI 16 env","SpCo\nTransfer'20\n0 env","SpCo\nTransfer'20\n1 env","SpCo\nTransfer'20\n2 env","SpCo\nTransfer'20\n4 env","SpCo\nTransfer'20\n8 env","SpCo\nTransfer'20\n16 env","SpCo\nTransfer'20\n+MI 0 env","SpCo\nTransfer'20\n+MI 1 env","SpCo\nTransfer'20\n+MI 2 env","SpCo\nTransfer'20\n+MI 4 env","SpCo\nTransfer'20\n+MI 8 env","SpCo\nTransfer'20\n+MI 16 env"]
colors = ["grey","grey","cyan","cyan","navajowhite","sandybrown","coral","tomato","orangered","red","navajowhite","sandybrown","coral","tomato","orangered","red"]
dic_num = 15
cl = np.array([1.0/dic_num for i in range(len(x))])

#x = []
y = []
e = []

for model in models:
    #f = "gibbs_result/similar_result/"+model+"/"+fn+".txt"
    fn = "name_acc_general"
    f = "gibbs_result/sigverse_result/"+model+"/Name_evaluation/"+fn+".txt"
    data_all = np.loadtxt(f,delimiter=" ")

    Mean = mean(data_all)
    Error = stdev(data_all)
    #Error = cilen(data_all)

    print(model+" "*(11-len(model))+":"+repr(np.round(Mean,3))+"+-"+repr(np.round(Error,3)))
    
    y.append(Mean)
    e.append(Error)

y = np.array(y)
e = np.array(e)

error_bar_set = dict(lw = 1, capthick = 1, capsize = 10)

fig = plt.figure(figsize=(17.5,5))
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)
bars = ax.bar(x, y,yerr = e,color=colors,error_kw=error_bar_set,edgecolor='black')
ax.set_ylabel('accuracy',fontsize=14)
#ax.plot(x, cl, label="chance level", color='black', linestyle="dashed")
plt.hlines(cl, -0.5, 15.5, label='chance level', color='black', linestyles='dashed')
plt.legend(fontsize=12,loc='upper left')
plt.ylim(-0.05,1.05)
plt.setp(ax.get_xticklabels(), fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=12)
#plt.show()
patterns = (None, '//', None,'//', None, None, None, None, None, None,'//', '//', '//', '//', '//', '//')
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)
plt.savefig("gibbs_result/"+result+"/Name_prediction_result/generalization/name_acc_general.png",dpi=1000)