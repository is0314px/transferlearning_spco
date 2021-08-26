import numpy as np
import shutil

LN = 200
TN = 100

idx_list = []
for i in range(LN):
    f = np.loadtxt("name_label_local/word"+repr(i)+".txt",dtype='S')
    #print(f)
    if f == "Emma's-room" or f == "mother's-room" or f == "father's-room":
        idx_list.append(i)

print(idx_list)

for i, j in enumerate(idx_list):
    k = LN+TN+i
    #print(j,k)
    shutil.copyfile("image/"+repr(j)+".png", "image/"+repr(k)+".png")
    shutil.copyfile("position/"+repr(j)+".txt", "position/"+repr(k)+".txt")
    shutil.copyfile("vision_fc7_normalized/"+repr(j)+".csv", "vision_fc7_normalized/"+repr(k)+".csv")
    try:
        shutil.copyfile("sentence/per_20/sentence"+repr(j)+".txt", "sentence/per_20/sentence"+repr(k)+".txt")
    except:
        pass
    
    try:
        shutil.copyfile("sentence/per_40/sentence"+repr(j)+".txt", "sentence/per_40/sentence"+repr(k)+".txt")
    except:
        pass
    
    try:
        shutil.copyfile("sentence/per_60/sentence"+repr(j)+".txt", "sentence/per_60/sentence"+repr(k)+".txt")
    except:
        pass
    
    try:
        shutil.copyfile("sentence/per_80/sentence"+repr(j)+".txt", "sentence/per_80/sentence"+repr(k)+".txt")
    except:
        pass
    
    try:
        shutil.copyfile("sentence/per_100/sentence"+repr(j)+".txt", "sentence/per_100/sentence"+repr(k)+".txt")
    except:
        pass
