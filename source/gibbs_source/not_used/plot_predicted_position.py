#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from pylab import *
import sys

result = sys.argv[1]

models = ["env_num_16"]
percent = [0,20,40,60,80,100]
rooms = ["Emma","mother","father"]

Positions = []

for model in models:
    pos_model = []

    for per in percent:
        pos_per = []

        for room in rooms:
            pos_xy = []

            f = "gibbs_result/sigverse_result/"+model+"/pos_"+room+".txt"
            data = np.loadtxt(f,delimiter=" ")

            for i in range(len(data)):
                if data[i][0] == per:
                    pos_xy.append([data[i][1],data[i][2]])
            
            pos_per.append(pos_xy)
        
        pos_model.append(pos_per)

    Positions.append(pos_model)


model_idx = [0]
for i in model_idx:
    for j in percent:
        l = j/20 #名前の付与率の刻み幅で割る
        fig, ax = plt.subplots()
        ax.invert_xaxis()
        img = Image.open("map/3ldk_9_local.PNG") 
        x_min = -590
        x_max =  590
        y_min = -470
        y_max =  470
        plt.imshow(img,extent=(x_min,x_max,y_min,y_max))
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        plt.xlabel('x',fontsize=14)
        plt.ylabel('y',fontsize=14)
        #ax.set_xticks([-500,-400,-300,-200,-100,0,100,200,300,400,500])
        #ax.set_yticks([-400,-300,-200,-100,0,100,200,300,400])
        #ax.tick_params(labelbottom="false",bottom="false") # x軸の削除
        #ax.tick_params(labelleft="false",left="false") # y軸の削除
        #ax.set_xticklabels([]) 
        #ax.set_yticklabels([]) 
        #box("false") #枠線の削除
   
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for k in range(20):
            x0.append(Positions[i][l][0][k][0])
            y0.append(Positions[i][l][0][k][1])
            x1.append(Positions[i][l][1][k][0])
            y1.append(Positions[i][l][1][k][1])
            x2.append(Positions[i][l][2][k][0])
            y2.append(Positions[i][l][2][k][1])

        plt.scatter(x0,y0,c = 'blue',marker = 'o',s = 50,label="Emma's-room",alpha=0.6)
        plt.scatter(x1,y1,c = 'red',marker = 'x',s = 50,label="mother's-room",alpha=0.6)
        plt.scatter(x2,y2,c = 'green',marker = '*',s = 50,label="father's-room",alpha=0.6)
        #plt.scatter(x1,y1,c = 'red',marker = 'x',s = 50,label="mother's-room")
        #plt.scatter(x0,y0,c = 'blue',marker = '.',s = 50,label="Emma's-room")

        plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.savefig("gibbs_result/"+result+"/Position_prediction_result/plotted_map/"+models[i]+"_per"+repr(j)+".png",dpi=1000)

#"""
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
#ax.tick_params(labelbottom="false",bottom="false") # x軸の削除
#ax.tick_params(labelleft="false",left="false") # y軸の削除
#ax.set_xticklabels([]) 
#box("false") #枠線の削除

#x = [1.75*62+365,1.75*62+365,4.5*62+365,4.5*62+365]
#y = [6*62+445,0.58*62+445,0.58*62+445,6*62+445]
x = [-590,-590,-50,-50]
y = [-160,-470,-470,-160]
plt.fill(x,y,color='blue',alpha=0.4,label="Emma's-room")
plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)

x = [-590,-590,-200,-200]
y = [330,-160,-160,330]
plt.fill(x,y,color='red',alpha=0.4,label="mother's-room")
plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)

x = [100,100,150,150,590,590]
y = [-160,-350,-350,-480,-480,-160]
plt.fill(x,y,color='green',alpha=0.4,label="father's-room")
plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)

plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.savefig("gibbs_result/"+result+"/Position_prediction_result/plotted_map/ground_truth_areas.png", dpi=1000)
#"""