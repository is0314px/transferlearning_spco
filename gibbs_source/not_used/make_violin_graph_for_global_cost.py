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
models = ["spcoa-LM","spcoa-LM+MI","env_num_0","env_num_1","env_num_2","env_num_4","env_num_8","env_num_16"]
x = ["SpCoA","SpCoA\n+MI","Trans\n0 env","Trans\n1 env","Trans\n2 env","Trans\n4 env","Trans\n8 env","Trans\n16 env"]
colors = ["grey","silver","navajowhite","sandybrown","coral","tomato","orangered","red"]

fig = plt.figure()
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)

df = pd.DataFrame({})
for idx, model in enumerate(models):
    #f = "gibbs_result/similar_result/"+model+"/"+fn+".txt"
    fn = "pos_cost_global"
    f = "gibbs_result/sigverse_result/"+model+"/"+fn+".txt"
    data_all = np.loadtxt(f,delimiter=" ")
    df[x[idx]] = data_all

df_melt = pd.melt(df)
sns.violinplot(x='variable',y='value', data=df_melt, jitter=False, palette=colors, ax=ax)
ax.set_xlabel(' ', fontsize=14)
ax.set_ylabel('cost', fontsize=14)
plt.ylim(0,1000)
plt.setp(ax.get_xticklabels(), fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=12)
#plt.show()
plt.savefig("gibbs_result/"+result+"/Position_prediction_result/generalization/pos_cost_global_violin.png",dpi=1000)
#"""