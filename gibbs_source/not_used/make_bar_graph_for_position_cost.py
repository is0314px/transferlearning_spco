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

#models = ["spcoa","env_num_2","env_num_8","env_num_32"]
models = ["spcoa","spcoa+MI","env_num_0","env_num_1","env_num_2","env_num_4","env_num_8","env_num_16"]
x = ["SpCoA","SpCoA\n+MI","Trans\n0 env","Trans\n1 env","Trans\n2 env","Trans\n4 env","Trans\n8 env","Trans\n16 env"]
colors = ["grey","silver","navajowhite","sandybrown","coral","tomato","orangered","red"]

y = []
e = []

for model in models:
    #f = "gibbs_result/similar_result/"+model+"/"+fn+".txt"
    fn = "pos_cost_general_ave"
    #fn = "cost_global"
    f = "gibbs_result/sigverse_result/"+model+"/"+fn+".txt"
    data_all = np.loadtxt(f,delimiter=" ")

    Mean = mean(data_all)
    Error = cilen(data_all)

    y.append(Mean)
    e.append(Error)

y = np.array(y)
e = np.array(e)

error_bar_set = dict(lw = 1, capthick = 1, capsize = 10)

fig = plt.figure()
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)
ax.bar(x, y, yerr = e,color=colors, error_kw=error_bar_set)
ax.set_ylabel('cost', fontsize=16)
plt.ylim(0,500)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=14)
#plt.show()
plt.savefig("gibbs_result/"+result+"/Position_prediction_result/generalization/position_cost_general_bar.png",dpi=1000)