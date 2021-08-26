#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from pylab import *
import sys

result = sys.argv[1]
env = '1DK_8' #check
dir = '../gibbs_source/map/'+env
range_txt = dir+'.txt'

range_pos = np.loadtxt(dir+'.txt',delimiter = " #") 

fig, ax = plt.subplots()
ax.invert_xaxis()
img = Image.open(dir+".PNG") 
plt.imshow(img)
x_min = range_pos[0]
x_max = range_pos[1]
y_min = range_pos[2]
y_max = range_pos[3]
plt.imshow(img,extent=(x_min,x_max,y_min,y_max))
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
#ax.tick_params(labelbottom="false",bottom="false") # x軸の削除
#ax.tick_params(labelleft="false",left="false") # y軸の削除
#ax.set_xticklabels([]) 
#box("false") #枠線の削除

"""
x = [-590,-590,-50,-50]
y = [-160,-470,-470,-160]
plt.fill(x,y, facecolor='blue', edgecolor='black', hatch="O", alpha=0.3, label="Emma's-room")
#plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)

x = [-590,-590,-200,-200]
y = [330,-160,-160,330]
plt.fill(x,y, facecolor='red', edgecolor='black', hatch="xx", alpha=0.3,label="mother's-room")
#plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)

x = [100,100,150,150,590,590]
y = [-160,-360,-360,-480,-480,-160]
plt.fill(x,y, facecolor='green', edgecolor='black', hatch="*", alpha=0.3, label="father's-room")
#plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0, fontsize=8)
#"""

#-----3LDK-----
#x = [-50,-50,150,150]
#y = [-360,-480,-480,-360]
#-----2LDK-----
#x = [-300,-300,-130,-130]
#y = [-10,-180,-180,-10]
#-----1LDK-----
#x = [-140,-140,140,140]
#y = [-90,-290,-290,-90]
#-----1DK-----
x = [250,250,450,450]
y = [-90,-300,-300,-90]

plt.fill(x,y, facecolor='cyan', edgecolor='black', hatch="\\\\", alpha=0.3,label="entrance")

#-----3LDK-----
#x = [-200,-200,-50,-50]
#y = [0,-160,-160,0]
#-----2LDK-----
#x = [-90,-90,0,0]
#y = [-340,-540,-540,-340]
#-----1LDK-----
#x = [140,140,240,240]
#y = [-90,-290,-290,-90]
#-----1DK-----
x = [100,100,250,250]
y = [-90,-300,-300,-90]

plt.fill(x,y, facecolor='magenta', edgecolor='black', hatch="//", alpha=0.3,label="toilet")

#-----3LDK-----
#x = [350,350,590,590]
#y = [0,-160,-160,0]
#-----2LDK-----
#x = [-300,-300,-90,-90]
#y = [-340,-540,-540,-340]
#-----1LDK-----
#x = [210,210,450,450]
#y = [-290,-500,-500,-290]
#-----1DK-----
x = [250,250,400,400]
y = [300,100,100,300]

plt.fill(x,y, facecolor='yellow', edgecolor='black', hatch="oo", alpha=0.3,label="bath")

#-----3LDK-----
#x = [350,350,590,590]
#y = [330,0,0,330]
#-----2LDK-----
#x = [100,100,450,450]
#y = [550,250,250,550]
#-----1LDK-----
#x = [-60,-60,220,220]
#y = [550,60,60,550]

#plt.fill(x,y, facecolor='mediumspringgreen', edgecolor='black', hatch="\\\\\\\\", alpha=0.3,label="living-room")

#-----3LDK-----
#x = [60,60,300,300]
#y = [330,140,140,330]
#-----2LDK-----
#x = [180,180,450,450]
#y = [200,-35,-35,200]
#-----1LDK-----
#x = [290,290,550,550]
#y = [550,240,240,550]
#-----1DK-----
x = [-200,-200,100,100]
y = [-300,-90,-90,-300]

plt.fill(x,y, facecolor='black', edgecolor='black', hatch="////", alpha=0.3,label="dining")

#-----3LDK-----
#x = [-200,-200,60,60]
#y = [330,50,50,330]
#-----2LDK-----
#x = [-140,-140,100,100]
#y = [550,170,170,550]
#-----1LDK-----
#x = [290,290,550,550]
#y = [240,-90,-90,240]
#-----1DK-----
x = [-200,-200,100,100]
y = [300,100,100,300]

plt.fill(x,y,facecolor='navy',  edgecolor='black', hatch="..", alpha=0.3,label="kitchen")

#-----3LDK-----
#x = [150,150,100,100,350,350]
#y = [0,-70,-70,-160,-160,0]
#-----2LDK-----
#x = [-300,-300,-90,-90]
#y = [-180,-340,-340,-180]
#-----1LDK-----
#x = [240,240,550,550]
#y = [-90,-290,-290,-90]
#-----1DK-----
x = [100,100,250,250]
y = [300,100,100,300]

plt.fill(x,y, facecolor='orangered',  edgecolor='black', hatch="||", alpha=0.3,label="washroom")

#-----3LDK-----
#x = [-590,-590,-50,-50]
#y = [-160,-470,-470,-160]
#-----2LDK-----
#x = [-450,-450,-140,-140]
#y = [550,-10,-10,550]
#-----1LDK-----
#x = [-500,-500,-550,-550,-140,-140,-60,-60]
#y = [300,200,200,-290,-290,0,0,300]
#-----1DK-----
x = [-500,-500,-200,-200]
y = [300,-300,-300,300]

plt.fill(x,y, facecolor='white',  edgecolor='black', hatch="x", alpha=0.3,label="bedroom")
plt.legend(bbox_to_anchor=(1, 1.1), loc='center right', borderaxespad=0, fontsize=8,ncol=3)

#-----3LDK-----
#x = [-590,-590,-200,-200]
#y = [330,-160,-160,330]
#-----2LDK-----
#x = [100,100,0,0,450,450]
#y = [-35,-340,-340,-540,-540,-35]

#plt.fill(x,y, facecolor='white',  edgecolor='black', hatch="x", alpha=0.3,label="bedroom")

"""
#-----3LDK-----
x = [100,100,150,150,590,590]
y = [-160,-360,-360,-480,-480,-160]
plt.fill(x,y, facecolor='white',  edgecolor='black', hatch="x", alpha=0.3,label="bedroom")
"""


plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.savefig("../gibbs_source/gibbs_result/"+result+"/Position_prediction_result/plotted_map/ground_truth_areas_.png", dpi=1000)
#"""