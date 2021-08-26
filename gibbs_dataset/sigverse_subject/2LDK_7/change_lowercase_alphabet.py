import matplotlib.pylab as plt
#import matplotlib
import numpy as np
import subprocess
from PIL import Image
import matplotlib.image as mpimg
from pylab import *

env_para = np.genfromtxt("Environment_parameter.txt", dtype = None, delimiter = " ")

DATA_NUM = int(env_para[6][1])+1

for i in range(DATA_NUM):

    fn = "sentence/per_100_/sentence"+ repr(i) + ".txt"
    f = open(fn, 'r')
    data = f.read()
    data = data.lower()
    print(repr(i)+" "+repr(data))
    f.close()

    """
    fn = "sentence/per_100/sentence"+ repr(i) + ".txt"
    f = open(fn, 'w')
    f.write(data)
    f.close()
    """