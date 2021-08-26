import os

data_num = 200 #max data number
each_num = 20 #the same name for each number
start_num = 0

dir = '../gibbs_dataset/sigverse/3LDK_9/'
name_dir = dir+'name_label_specific/'
sentence_dir = dir+'sentence_label_specific/'
"""
try:
    os.mkdir(dir+'name_label')
    os.mkdir(dir+'sentence_label')
    os.mkdir(dir+'name')
    os.mkdir(dir+'sentence')
except:
    pass
"""

for j in range(data_num):
    i = j + start_num

    if j % each_num == 0 and (i == 20 or i == 40 or i== 180):
        text = raw_input("name between {}-{}? >> ".format(i, i+each_num-1))
    
    try:
        with open(name_dir+"word"+repr(i)+".txt","w") as f:
            f.write(text + " ")
        #print(text + " ")
        
        if i % 5 == 0:
            with open(sentence_dir+"sentence"+repr(i)+".txt","w") as f:
                f.write("this is the "+text+" ") #check!
                #f.write("this is "+text+" ")
            #print("this is the "+text+" ")

        elif i % 5 == 1:
            with open(sentence_dir+"sentence"+repr(i)+".txt","w") as f:
                f.write("the "+text+" is here ") #check!
                #f.write(text+" is here ")
            #print("the "+text+" is here ")

        elif i % 5 == 2:
            with open(sentence_dir+"sentence"+repr(i)+".txt","w") as f:
                f.write("this place is the "+text+" ") #check!
                #f.write("this place is "+text+" ")
            #print("this place is the "+text+" ")

        elif i % 5 == 3:
            with open(sentence_dir+"sentence"+repr(i)+".txt","w") as f:
                f.write("this space is the "+text+" ") #check!
                #f.write("this space is "+text+" ")
            #print("this space is the "+text+" ")

        elif i % 5 == 4:
            with open(sentence_dir+"sentence"+repr(i)+".txt","w") as f:
                f.write("this location is the "+text+" ") #check!
                #f.write("this location is "+text+" ")
            #print("this location is the "+text+" ")
    except:
        pass