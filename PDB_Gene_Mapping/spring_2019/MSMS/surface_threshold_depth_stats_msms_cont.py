# -*- coding: utf-8 -*-
"""
Created on %19-Feb-2019(10.33 A.m)

@author: %A.Nishanth C00294860
"""

import os
import pickle
import numpy as np
def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

#%%
""" round the depths to first digits have a numpy to store the values between 0-9
 row specify the number and column sepcify the first digit
for example 3.4 means coungt added to 3rd row and 4 th digit 
"""
array_count_ca = np.zeros((11,10))
array_count_res = np.zeros((11,10))

def array_index_helper(depth_temp):
    if round(depth_temp,1)>10.9:
        row=10
        column=9
    elif round(depth_temp,1)>9.9:
        row=10
        column=int(str(round(depth_temp,1))[3])
    else:
        row=int(str(round(depth_temp,1))[0])
        column=int(str(round(depth_temp,1))[2])
    return row, column   
#%%
def array_count_updater(pdb_name,array_count_ca,array_count_res,loading_MOTIF_dir):
    """
    This function update the counter number of occurfance of depth in numpy arrays
    """    
    os.chdir('/')
    os.chdir(loading_MOTIF_dir)
    depth_ca_temp =pickle.load(open( ''.join(["depth_ca_temp_",pdb_name,".p"]), "rb" ))
    depth_res_temp =pickle.load(open( ''.join(["depth_res_temp_",pdb_name,".p"]), "rb" ))
    for i in range(0,len(depth_ca_temp)):
        for j in range(0,len(depth_ca_temp[i])):
            row, column = array_index_helper(depth_ca_temp[i][j])
            array_count_ca[row][column]=array_count_ca[row][column]+1
    
            row_2, column_2 = array_index_helper(depth_res_temp[i][j])
            array_count_res[row_2][column_2]=array_count_res[row_2][column_2]+1
            
    return array_count_ca,array_count_res

def threshold_define(array_count_ca,array_count_res):
    """
    first take sum and check the which is 7,8,9,10,11%s
    since  sum_count for ca and residue same
    
    it returns the threshold values for ca and residue depth from the surface
    which will loose values from 7%,8%,9%,10%,11% accordingly
    """
    sum_count_ca = np.sum(array_count_ca)        
#    percentage_indexes=[int(round(sum_count_ca*0.04)),int(round(sum_count_ca*0.05)),int(round(sum_count_ca*0.06)),int(round(sum_count_ca*0.07)),int(round(sum_count_ca*0.08)),int(round(sum_count_ca*0.09)),int(round(sum_count_ca*0.10))]       
    percentage_indexes=[int(round(sum_count_ca*0.07)),int(round(sum_count_ca*0.08)),int(round(sum_count_ca*0.09)),int(round(sum_count_ca*0.10)),int(round(sum_count_ca*0.11))]       

    #% summing the indexes from the backwards and choose the threshold
    chk_ca = 0
    chk_res = 0
    k1=0
    k2=0
    thresh_ca=[]
    thresh_res=[]
    for i in range(10,0,-1):
        for j in range(9,0,-1):
            chk_ca = array_count_ca[i][j]+ chk_ca
            chk_res = array_count_res[i][j]+ chk_res
            if len(percentage_indexes)>k1 and chk_ca>= percentage_indexes[k1]:
                thresh_ca.append([i,j])
                k1=k1+1
            if len(percentage_indexes)>k2 and chk_res>= percentage_indexes[k2]:
                thresh_res.append([i,j])
                k2=k2+1
    return thresh_ca, thresh_res
#%%
array_count_ca = np.zeros((11,10))
array_count_res = np.zeros((11,10))

name = 'ONGO'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        if pdb_name!='4MDQ':
#            print("PDB in progress: ", pdb_name)
            array_count_ca,array_count_res = array_count_updater(pdb_name,array_count_ca,array_count_res,loading_MOTIF_dir)
#%%
#array_count_ca = np.zeros((11,10))
#array_count_res = np.zeros((11,10))

name = 'TSG'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
#            print("PDB in progress: ", pdb_name)
            array_count_ca,array_count_res = array_count_updater(pdb_name,array_count_ca,array_count_res,loading_MOTIF_dir)
##%%
#array_count_ca = np.zeros((11,10))
#array_count_res = np.zeros((11,10))

name = 'Fusion'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
#            print("PDB in progress: ", pdb_name)
            array_count_ca,array_count_res = array_count_updater(pdb_name,array_count_ca,array_count_res,loading_MOTIF_dir)
        
thresh_ca_list, thresh_res_list = threshold_define(array_count_ca,array_count_res)
#%%
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
os.chdir('/')
os.chdir(loading_pikle_dir_part)
pickle.dump(thresh_ca_list, open("thresh_ca_list.p", "wb" ) ) 
pickle.dump(thresh_res_list, open("thresh_res_list.p", "wb" ) ) 
pickle.dump(array_count_ca, open("array_count_ca.p", "wb" ) ) 
pickle.dump(array_count_res, open("array_count_res.p", "wb" ) ) 
#%%

def pdb_thresh_sat_count(pdb_name,thresh_change_sat_count,thresh_ca_list,thresh_res_list):
    """
    Then varying the thresold among the each direction and check howmany C_A atoms
    satisfied 
    
    """
    os.chdir('/')
    os.chdir(loading_MOTIF_dir)
    depth_ca_temp =pickle.load(open( ''.join(["depth_ca_temp_",pdb_name,".p"]), "rb" ))
    depth_res_temp =pickle.load(open( ''.join(["depth_res_temp_",pdb_name,".p"]), "rb" ))
    
    for k in range(0,len(thresh_ca_list)):
        thresh_ca = float(''.join([str(thresh_ca_list[k][0]),'.',str(thresh_ca_list[k][1])]))
        for m in range(0,len(thresh_res_list)):
            thresh_res = float(''.join([str(thresh_res_list[m][0]),'.',str(thresh_res_list[m][1])]))
    #    print(thresh_ca)
            for i in range(0,len(depth_ca_temp)):
                for j in range(0,len(depth_ca_temp[i])):
                    if depth_ca_temp[i][j]<=thresh_ca:
                        if depth_res_temp[i][j]<=thresh_res:
                            thresh_change_sat_count[k][m] = thresh_change_sat_count[k][m]+1
        
    return thresh_change_sat_count



os.chdir('/')
os.chdir(loading_pikle_dir_part)
""" checking the threshold conditions """
thresh_ca_list =pickle.load(open("thresh_ca_list.p", "rb" ))
thresh_res_list =pickle.load(open("thresh_res_list.p", "rb" ))

#%%
thresh_change_sat_count = np.zeros((len(thresh_ca_list),len(thresh_res_list)))

name = 'ONGO'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        if pdb_name!='4MDQ':
            thresh_change_sat_count = pdb_thresh_sat_count(pdb_name,thresh_change_sat_count,thresh_ca_list,thresh_res_list)

name = 'TSG'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        thresh_change_sat_count = pdb_thresh_sat_count(pdb_name,thresh_change_sat_count,thresh_ca_list,thresh_res_list)

name = 'Fusion'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        thresh_change_sat_count = pdb_thresh_sat_count(pdb_name,thresh_change_sat_count,thresh_ca_list,thresh_res_list)
#%% then calculate the percentages of missing
thresh_per =np.zeros((len(thresh_ca_list),len(thresh_res_list)))
sum_count_ca = np.sum(array_count_ca)        
for i in range(0,len(thresh_ca_list)):
    for j in range(0,len(thresh_res_list)):
        thresh_per[i][j]=100*(sum_count_ca-thresh_change_sat_count[i][j])/sum_count_ca
#%%
str_thres_ca=[]
str_thres_res=[]
for k in range(0,len(thresh_ca_list)):
    str_thres_ca.append(''.join([str(thresh_ca_list[k][0]),'.',str(thresh_ca_list[k][1])]))
for m in range(0,len(thresh_res_list)):
    str_thres_res.append(''.join([str(thresh_res_list[m][0]),'.',str(thresh_res_list[m][1])]))

#%%
'''
From the results obtained above check the percentage of C_alpha carbon covered as surface
#his will check upto 25% of MOTIF atom loss while selecting surface atoms
'''
checking_thresh_ca=[7.2,6.9,6.7]
checking_thresh_res=[6.9,6.7,6.6]
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

def sur_per_saver(pdb_name,loading_pikle_dir,saving_dir):
    sat_ca_sur_thresh =np.zeros((3,3))
    
    os.chdir('/')
    os.chdir(loading_pikle_dir)
#    coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
    fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
    fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
    for i in range(0,len(checking_thresh_ca)):
        for j in range(0,len(checking_thresh_res)):
            for k in range(0,len(fin_ca_depth_all)):
                if fin_ca_depth_all[k]<=checking_thresh_ca[i]:
                    if fin_res_depth_all[k]<=checking_thresh_res[j]:
                        sat_ca_sur_thresh[i][j]=sat_ca_sur_thresh[i][j]+1
    for i in range(0,len(checking_thresh_ca)):
        for j in range(0,len(checking_thresh_res)):
            sat_ca_sur_thresh[i][j]=100*sat_ca_sur_thresh[i][j]/len(fin_ca_depth_all)
    
    np.save(''.join(['sat_ca_sur_thresh_',pdb_name,'.npy']), sat_ca_sur_thresh) 
#%%
name = 'ONGO'
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
saving_dir=''.join([loading_pikle_dir_part,name,"/c_a_sur_per/"])
name = 'ONGO'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        if pdb_name!='4MDQ':
            print("PDB in progress: ", pdb_name)
            sur_per_saver(pdb_name,loading_pikle_dir,saving_dir)
#%%
'''After defining the threshold check howmuch percentage of 
MOTIF-C_alpha atoms missed for each group'''
checking_thresh_ca=[7.2,6.9,6.7]
checking_thresh_res=[6.9,6.7,6.6]
'''repeating commands'''
array_count_ca = np.zeros((11,10))
array_count_res = np.zeros((11,10))

name = 'ONGO'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        if pdb_name!='4MDQ':
#            print("PDB in progress: ", pdb_name)
            array_count_ca,array_count_res = array_count_updater(pdb_name,array_count_ca,array_count_res,loading_MOTIF_dir)
#%%
array_count_sat_temp = np.zeros((11,10))
for i in range(0,np.size(array_count_ca,0)):
    for j in range(0,np.size(array_count_ca,1)):
        if float(''.join([str(i),'.',str(j)]))<=checking_thresh_ca[k]:        

        #%%
i=0
j=1
#%%
''' checking purpose       '''
name = 'ONGO'
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_name = '3QKK'
array_count_ca = np.zeros((11,10))
array_count_res = np.zeros((11,10))

"""
This function update the counter number of occurfance of depth in numpy arrays
"""    
os.chdir('/')
os.chdir(loading_MOTIF_dir)
depth_ca_temp =pickle.load(open( ''.join(["depth_ca_temp_",pdb_name,".p"]), "rb" ))
depth_res_temp =pickle.load(open( ''.join(["depth_res_temp_",pdb_name,".p"]), "rb" ))
#for i in range(0,len(depth_ca_temp)):
#    for j in range(0,len(depth_ca_temp[i])):
#        print('i: ',i,' j: ',j)
#        row, column = array_index_helper(depth_ca_temp[i][j])
#        array_count_ca[row][column]=array_count_ca[row][column]+1
#
#        row_2, column_2 = array_index_helper(depth_res_temp[i][j])
#        array_count_res[row_2][column_2]=array_count_res[row_2][column_2]+1
