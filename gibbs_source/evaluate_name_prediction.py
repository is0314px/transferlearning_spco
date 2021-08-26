#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import sys

def test_data_read(file,test_num):#read number of test data
    i = 0
    test_data_num = []
    for line in open(file, 'r').readlines(): # 0,1,2,3,8,9,...
        
        if i == test_num: # line's row number == test_num
            num = line[:-1].split(',') # make number list and delete \n  [0,1,2,3,8,9]

            for n in num:
                try:
                    test_data_num.append(int(n))
                except ValueError:
                    pass
        i += 1
    
    return test_data_num

if __name__ == "__main__":

    train_list = sys.argv[1]
    result_dir = sys.argv[2] 
    dic_txt = sys.argv[3]
    ex_type = sys.argv[4]
    room_type = sys.argv[5]

    name_dic = np.loadtxt("../gibbs_dataset/dataset_path_txt/"+dic_txt, delimiter="\n", dtype="S" )
    Training_list = np.genfromtxt(train_list , delimiter="\n", dtype='S' )

    try:
        test_dataset_index = len(Training_list)-1
    except:
        Training_list = np.array([Training_list])
        test_dataset_index = 0

    dataset_dir = Training_list[int(test_dataset_index)]

    per = 0
    PER = 20

    if ex_type == "general":
        MAX_PER = 0
        test_num_txt = dataset_dir + "/test_num.txt"
        label_dir = dataset_dir + "/name_label"
    elif ex_type == "specific":
        MAX_PER = 100 
        test_num_txt = dataset_dir + "/test_num_specific.txt"
        label_dir = dataset_dir + "/name_label_specific"
    else:
        print("ERROR")
        sys.exit()

    while per <= MAX_PER:
        predicted_txt = result_dir+"/per_"+repr(per)+"_iter_200/dataset"+repr(int(test_dataset_index))+"/predicted_names.txt"
        test_num = 0
        test_data_num = test_data_read(test_num_txt,test_num) #read test data number
        
        label_name = []
        for test_data_index in test_data_num:
            temp = np.loadtxt(label_dir+"/word"+repr(test_data_index)+".txt", delimiter="\n", dtype="S" ) #read label name data
            label_name.append(temp)

        read_predicted_txt = np.loadtxt(predicted_txt, delimiter=" ", dtype="S" ) #read predicted name data

        N = 0 #0,1,2,3,..., all label number
        cnt = 0 #count matching number
        est_name_num = [0 for i in range(len(name_dic))]
        label_name_num = [0 for i in range(len(name_dic))]
        prob_k = [0 for i in range(len(name_dic))]
        cnt_vector = [0 for i in range(len(name_dic))]

        for predicted_name in read_predicted_txt:
            
            if int(predicted_name[0]) in test_data_num:
                #print(str(predicted_name[0])+' '+str(predicted_name[1])+' '+str(label_name[N]))
                temp = str(label_name[N]).replace(' ',"")

                if str(predicted_name[1])  == temp: #if predicted name matched to label name, cnt is incremented
                    cnt += 1
                    
                    name = label_name[N].tolist().replace(' ','')
                    cnt_vector[name_dic.tolist().index(name)] += 1

                """
                for k, names in enumerate(name_dic):
                    if names == str(predicted_name[1]):
                        est_name_num[k] += 1
                    if names in str(label_name[N]):
                        label_name_num[k] += 1
                """
                
                N = N+1

        """
        chance_prob = 0
        for k in range(len(name_dic)):
            prob_k[k] = (float(est_name_num[k])/float(N)) * (float(label_name_num[k])/float(N))
            chance_prob += prob_k[k]
        """
        
        accuracy = float(cnt) / float(N)
        #kappa = (accuracy-chance_prob) / (1-chance_prob) #caluculae kappa statistic
        max_width = max(len(s) for s in name_dic)

        if ex_type == "general":
            print("name accuracy")
            print("name teaching rate "+repr(per))
            for n, name in enumerate(name_dic):
                if n >= 7: #entrance ~ 
                    if n == 9: #bedroom
                        if room_type == '1LDK':
                            data_num_each = 10
                        elif room_type == '2LDK':
                            data_num_each = 20
                        elif room_type == '3LDK':
                            data_num_each = 30
                    else:
                        data_num_each = 10
                    
                    acc_each = float(cnt_vector[n])/float(data_num_each)
                    print(name+" "*(max_width-len(name_dic[n]))+": "+repr(acc_each))
                    f = open(result_dir+'/Name_evaluation/name_acc_'+name+'.txt','a')
                    f.write(repr(acc_each)+'\n')
                    f.close()

            print("average"+" "*(max_width-7)+": "+repr(accuracy))
            print("-------------------")

            f = open(result_dir+'/Name_evaluation/name_acc_general.txt','a')
            f.write(repr(accuracy)+'\n')
            f.close()
        else:
            print("name accuracy")
            print("name teaching rate "+repr(per))
            for n, name in enumerate(name_dic):
                if n >= 15: #Emma's-room ~
                    acc_each = float(cnt_vector[n])/10
                    print(name+" "*(max_width-len(name_dic[n]))+": "+repr(acc_each))
                    f = open(result_dir+'/Name_evaluation/name_acc_'+name+'.txt','a')
                    f.write(repr(per)+" "+repr(acc_each)+'\n')
                    f.close()

            print("average"+" "*(max_width-7)+": "+repr(accuracy))
            print("-------------------")

            f = open(result_dir+'/Name_evaluation/name_acc_specific.txt','a')
            f.write(repr(per)+" "+repr(accuracy)+'\n')
            f.close()

        per = per + PER