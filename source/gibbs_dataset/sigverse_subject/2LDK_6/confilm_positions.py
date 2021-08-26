import matplotlib.pylab as plt
#import matplotlib
import numpy as np
import subprocess
from PIL import Image
import matplotlib.image as mpimg
from pylab import *

env_para = np.genfromtxt("Environment_parameter.txt", dtype = None, delimiter = " ")

x_max = float(env_para[0][1])*100 #-590
y_max = float(env_para[1][1])*100 #590
x_min = float(env_para[2][1])*100 #-470
y_min = float(env_para[3][1])*100 #470
DATA_NUM = int(env_para[6][1])+1
env_name = env_para[7][1] #environment name
env_name = env_name.decode('utf-8') #python 3
#env_name_py2 = str(env_name) #python 2

x = []
y = []

for i in range(DATA_NUM):

    f = "position/"+ repr(i) + ".txt"
    pos = []
    for line in open(f, 'r').readlines():
        data = line[:-1].split(' ')
        pos += [float(data[0].replace('\r', ''))]
        pos += [float(data[1].replace('\r', ''))]
    pos = [pos[0]*100,pos[1]*100,pos[2],pos[3]]

    x.append(float(pos[0]))
    y.append(float(pos[1]))

fig, ax = plt.subplots()
ax.invert_xaxis()
map = Image.open(env_name+".PNG")
plt.imshow(map,extent=(x_min,x_max,y_min,y_max))
plt.scatter(x,y,c = 'black',marker = '.',s = 50,alpha=1)
plt.show()