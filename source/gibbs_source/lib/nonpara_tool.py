#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math

#stick breaking process
def stick_breaking(gamma,Stick_large_L):
    v = []
    pi = np.array([])
    for k in range(Stick_large_L):
        v_k = np.random.beta(1,gamma)
        v.append(v_k)
        pi_k = v[-1]
        for i in range(k):
            pi_k *= (1-v[-(i+2)])        
  
        pi = np.append(pi,pi_k)    

    pi /= math.fsum(pi)
    return pi

#chinese restaurant process
def CRP_init():
    data_class_num = [1.0]
    class_num = 1
    return class_num

#def CRP_cal_prob(i,class_num,data_num,class_count):
def CRP_cal_prob(i,class_num,data_num,class_count,gamma):
    
    if i == (class_num-1):
        prob = gamma/(data_num-1+gamma)    
    else:
        prob = class_count/(data_num-1+gamma)
    return prob

#def CRP(phi,pi,):
