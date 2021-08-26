#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from pylab import *
import sys

env = '1DK_8'
dir = '../../gibbs_dataset/sigverse/'+env+'/'
position_dir = dir + 'position/'
img_dir = env+'.PNG'
x_min = -500
x_max =  500
y_min = -300
y_max =  300
N = 300 #読み込むデータ数の上限

def position_data_read(directory):
    
    all_position = [] 

    for i in range(N):
        try:
            f = directory + repr(i) + ".txt"
            position = [] #(x,y,sin,cos)

            for line in open(f, 'r').readlines():
                data=line[:-1].split(' ')
                position +=[float(data[0].replace('\r', ''))]
                position +=[float(data[1].replace('\r', ''))]

            position = [position[0]*100,position[1]*100,position[2],position[3]] #x and y multiplied by 100 注意！！
            all_position.append(position)

        except:
            pass

    return np.array(all_position)

fig, ax = plt.subplots()
ax.invert_xaxis()
img = Image.open(img_dir) 
plt.imshow(img,extent=(x_min,x_max,y_min,y_max))
ax.set_xlim(x_min,x_max)
ax.set_ylim(y_min,y_max)

position = position_data_read(position_dir)
for i in range(N):
    try:
        plt.plot(position[i,0],position[i,1],marker='.',color='blue')
    except:
        pass

plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.savefig('ploted_map.png',dpi=1000)