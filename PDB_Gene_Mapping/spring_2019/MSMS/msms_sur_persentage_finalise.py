# -*- coding: utf-8 -*-
"""
Created on %21-Feb-2019(05.40 p.m)

@author: %A.Nishanth C00294860
"""

import os
import pickle
import numpy as np
import copy
def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details
#%%
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

'''
From the results obtained above check the percentage of C_alpha carbon covered as surface
#his will check upto 25% of MOTIF atom loss while selecting surface atoms
'''
checking_thresh_ca=[7.2,6.9,6.7]
checking_thresh_res=[6.9,6.7,6.6]
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"


#%%
name = 'ONGO'
loading_pikle_dir=''.join([loading_pikle_dir_part,name,"/c_a_sur_per"])
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)

sat_ca_sur_thresh =np.zeros((3,3))
os.chdir('/')
os.chdir(loading_pikle_dir)
c=0
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        if pdb_name!='4MDQ':
#            print("PDB in progress: ", pdb_name)
            sat_ca_sur_thresh_temp = np.load(''.join(['sat_ca_sur_thresh_',pdb_name,'.npy'])) 
            sat_ca_sur_thresh = sat_ca_sur_thresh + sat_ca_sur_thresh_temp
            c=c+1

sat_ca_sur_thresh_ongo = copy.deepcopy(sat_ca_sur_thresh/c)            
#%%
name = 'TSG'
loading_pikle_dir=''.join([loading_pikle_dir_part,name,"/c_a_sur_per"])
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)

sat_ca_sur_thresh =np.zeros((3,3))
os.chdir('/')
os.chdir(loading_pikle_dir)
c=0
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        sat_ca_sur_thresh_temp = np.load(''.join(['sat_ca_sur_thresh_',pdb_name,'.npy'])) 
        sat_ca_sur_thresh = sat_ca_sur_thresh + sat_ca_sur_thresh_temp
        c=c+1
sat_ca_sur_thresh_tsg = copy.deepcopy(sat_ca_sur_thresh/c)            
#%%
name = 'Fusion'
loading_pikle_dir=''.join([loading_pikle_dir_part,name,"/c_a_sur_per"])
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)

sat_ca_sur_thresh =np.zeros((3,3))
os.chdir('/')
os.chdir(loading_pikle_dir)
c=0
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        sat_ca_sur_thresh_temp = np.load(''.join(['sat_ca_sur_thresh_',pdb_name,'.npy'])) 
        sat_ca_sur_thresh = sat_ca_sur_thresh + sat_ca_sur_thresh_temp
        c=c+1
sat_ca_sur_thresh_fusion = copy.deepcopy(sat_ca_sur_thresh/c)           
#%%
sat_ca_sur_thresh = (sat_ca_sur_thresh_ongo+sat_ca_sur_thresh_tsg+sat_ca_sur_thresh_fusion)/3