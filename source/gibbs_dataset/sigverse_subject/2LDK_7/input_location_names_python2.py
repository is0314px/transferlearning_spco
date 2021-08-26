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
#env_name = env_name.decode('utf-8') #python 3
env_name_py2 = str(env_name) #python 2

for i in range(DATA_NUM):
    #make image of visual information
    fig, ax = plt.subplots()
    ax.invert_xaxis()
    ax.axis("off")
    image = Image.open("image/"+repr(i)+".png") 
    plt.imshow(image)
    thismanager = get_current_fig_manager()
    thismanager.window.wm_geometry("+100+0")

    #make image of position information
    fig, ax = plt.subplots()
    ax.invert_xaxis()
    f = "position/"+ repr(i) + ".txt"
    pos = []
    for line in open(f, 'r').readlines():
        data = line[:-1].split(' ')
        pos += [float(data[0].replace('\r', ''))]
        pos += [float(data[1].replace('\r', ''))]
    pos = [pos[0]*100,pos[1]*100,pos[2],pos[3]]

    #map = Image.open("3ldk_9_local.PNG")
    map = Image.open(env_name+".PNG")
    plt.imshow(map,extent=(x_min,x_max,y_min,y_max))

    x = float(pos[0])
    y = float(pos[1])
    dx = float(pos[2])*75 #sin
    dy = float(pos[3])*75 #cos

    plt.scatter(x,y,c = 'black',marker = 'o',s = 50,alpha=1)
    plt.quiver(x,y,dx,dy,angles='xy',scale_units='xy',scale=1,color='black')

    thismanager = get_current_fig_manager()
    thismanager.window.wm_geometry("+100+600")

    plt.ion()

    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    plt.xlabel('x',fontsize=14)
    plt.ylabel('y',fontsize=14)

    plt.show()
    #name = input("Please tell me the location name ("+repr(i)+"/"+repr(DATA_NUM-1)+"): ") #python 3
    name = raw_input("Please tell me the location name ("+repr(i)+"/"+repr(DATA_NUM-1)+"): ") #python 2
    
    name_file = open("sentence/per_100/sentence"+repr(i)+".txt","w")
    name_file.write(name)
    name_file.close()

    plt.close()
    plt.close()
