import os
import sys
import numpy as np

dir = 'gibbs_result/sigverse_result/'
dic_txt = sys.argv[1]
model = sys.argv[2]
ex_type = sys.argv[3]

name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic_txt, delimiter = "\n", dtype = "S" ) #dictionary of name
file_list = []

for name in name_dic:
    file_list.append(name)

file_list.append(ex_type)

print(file_list)

for idx, file in enumerate(file_list):
    try:
        os.remove(dir+model+'/Name_evaluation/name_acc_'+file+'.txt')
    except:
        pass
    
    try:
        os.remove(dir+model+'/Position_evaluation/pos_acc_'+file+'.txt')
    except:
        pass
    
    try:
        os.remove(dir+model+'/Position_evaluation/pos_cost_'+file+'.txt')
    except:
        pass


