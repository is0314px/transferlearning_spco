#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import math
import random
import scipy.stats as ss
#stick breaking process
def stick_breaking(gamma,Stick_large_L):
    v = []
    pi = np.array([])
    all_sum = 0.0
    for k in range(Stick_large_L):
        a = np.random.beta(1,gamma)
        v.append(a)
        pi_k = v[-1] #リストvの最後を参照している(=a)
         
        for i in range(k):
            pi_k *= (1-v[-(i+2)])        
            
        all_sum = all_sum + pi_k

        pi = np.append(pi,pi_k)
        if all_sum == 1.0 or all_sum >= 0.999999999999:
            pi /= math.fsum(pi)
            return pi
    
    pi /= math.fsum(pi)
    return pi

def crp(alpha,n,rand_array): #alpha:パラメータ n:SBPによるpi
    count=len(n)
    #s=[] #各人が座るテーブル番号のリスト
    #table={} #テーブルごとの人数の辞書
    new_phi=np.array([0.0 for w in range(len(rand_array))])#新たに生成されるphi
    index_list=[]#選択されたindexを格納するlist
    index = np.arange(len(rand_array))#注意:実際のindexとrand_arrayは転移する概念のindex数と各々の値に合わせる
    custm = ss.rv_discrete(name='custm', values=(index, rand_array))#読み込んだ分布
    inc = 1.0/float(count) #増加分
    #sampled_rvs = custm.rvs(size=count)
    for i in range(count):
        if i==0:
            first_sampled_rvs = custm.rvs(size=1)
            index_list.append(first_sampled_rvs[0])
            #np.append(index_list,first_sampled_rvs[0])
            new_phi[first_sampled_rvs] += inc
            #print index_list
            #print first_sampled_rvs
            #print new_phi
        else:
            prob = random.random() #0-1の範囲
            sums = 0.0 #各テーブルの着席確率を累積していって、probを超えたら着席
            
            #新規テーブルに対して
            new_p = float(alpha)/(i-1+alpha)
            
            if(new_p>=prob):
                sampled_rvs = custm.rvs(size=1)
                while (sampled_rvs in index_list):
                    sampled_rvs = custm.rvs(size=1)
                index_list.append(sampled_rvs[0])
                #np.append(index_list,sampled_rvs)
                new_phi[sampled_rvs] += inc
            else:
                sums += new_p
                number=-1
                list_len=len(index_list)
                #print list_len
                while (prob>sums):
                    sums += 1.0/(i-1+alpha)
                    number = number+1
                    #print number
                if number >= list_len:
                    new_phi[index_list[list_len-1]]+=inc
                else:
                    new_phi[index_list[number]]+=inc
    return new_phi
'''
if __name__ == "__main__":
    s=stick_breaking(0.9,100)
    print s
    print len(s)
    print sum(s)
    rand_array = np.random.rand(20)
    rand_array/=math.fsum(rand_array)
    c=crp(0.5, s, rand_array)
    print c'''
