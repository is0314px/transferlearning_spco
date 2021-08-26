import matplotlib.pylab as plt
#import matplotlib
import numpy as np
import subprocess
from PIL import Image
import matplotlib.image as mpimg
from pylab import *

env_list = ["1LDK_1","1LDK_2","1LDK_3","1LDK_4","1LDK_5","1LDK_6","1LDK_7","1LDK_8","2LDK_1","2LDK_2","2LDK_3","2LDK_4","2LDK_5","2LDK_6","2LDK_7","2LDK_8"]
dic = []
#error_list = ["TV","If"]

for env in env_list:
    env_para = np.genfromtxt(env+"/Environment_parameter.txt", dtype = None, delimiter = " ")

    DATA_NUM = int(env_para[6][1])+1

    for i in range(DATA_NUM):

        fn = env+"/sentence/per_100/sentence"+ repr(i) + ".txt"
        f = open(fn, 'r')
        #data = f.read()
        data = np.loadtxt(f, delimiter=" ", dtype='S')
        data = data.tolist()

        #print(repr(env)+" "+repr(i)+" "+repr(data))
        print(data)

        if isinstance(data,list):
            #print('list')
            for d in data:
                #print(repr(env)+" "+repr(i)+" "+repr(d))
                #print(d)
                dic.append(d)

        elif isinstance(data,str):
            #print(repr(env)+" "+repr(i)+" "+repr(d))
            #print('str')
            dic.append(d)

        f.close()

dictionary = np.sort(list(set(dic)))

"""

for word in dictionary:
    fn = "name_dictionary_subject.txt"
    f = open(fn, 'a')
    f.write(word+"\n")
    f.close()
"""
    


#print(dictionary)

