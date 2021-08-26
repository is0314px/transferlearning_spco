#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math
#stick breaking process
def stick_breaking(gamma,Stick_large_L):
    v=[]
    pi=np.array([])
    all_sum=0.0
    for k in range(Stick_large_L):
        a=np.random.beta(1,gamma)
        v.append(a)
        pi_k=v[-1]#リストvの最後を参照している(=a)
        #print pi_k
        for i in range(k):
            pi_k*=(1-v[-(i+2)])        
            
        all_sum=all_sum+pi_k
        print all_sum
        pi=np.append(pi,pi_k)
        if all_sum == 1.0 or all_sum >= 0.999999999999:
            pi /= math.fsum(pi)
            return pi
    
    pi /=math.fsum(pi)
    return pi

#chinese restaurant process

def CRP_init():
    data_class_num=[1.0]
    class_num=1
    return class_num
#def CRP_cal_prob(i,class_num,data_num,class_count):
def CRP_cal_prob(i,class_num,data_num,class_count,gamma):
    
    if i==(class_num-1):
        prob=gamma/(data_num-1+gamma)    
    else:
        prob=class_count/(data_num-1+gamma)
    return prob

#def CRP(phi,pi,):
