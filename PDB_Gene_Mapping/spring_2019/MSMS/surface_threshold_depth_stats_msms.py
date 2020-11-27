# -*- coding: utf-8 -*-
"""
Created on %18-Feb-2019(12.29 P.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle
import numpy as np

def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
#    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
#%%
def pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir):
    """
    This function create a pikle files for MOTIF depths
    """
    os.chdir('/')
    os.chdir(loading_pikle_dir)
    coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
    fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
    fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
    # first verify the mapping MSMS is done correctly by the length of the coordinates and fin_.._deph all shold be same
    if len(coordinates)!=len(fin_ca_depth_all):
        print("problemtric_pdb: ",pdb_name) 
#        raise Exception("problemtric_pdb: ",pdb_name) 
    else:
        MOTIF_indexes_all =pickle.load(open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))
        
        depth_ca_temp=[]
        depth_res_temp=[]
        for i_list in MOTIF_indexes_all:
            depth_ca_t=[]
            depth_res_t=[]
            for i in i_list:
                depth_ca_t.append(fin_ca_depth_all[i])
                depth_res_t.append(fin_res_depth_all[i])
            depth_ca_temp.append(depth_ca_t)
            depth_res_temp.append(depth_res_t)
            
        os.chdir('/')
        os.chdir(saving_dir)
        pickle.dump(depth_ca_temp, open( ''.join(["depth_ca_temp_",pdb_name,".p"]), "wb" ) ) 
        pickle.dump(depth_res_temp, open( ''.join(["depth_res_temp_",pdb_name,".p"]), "wb" ) ) 
        
#%%
name = "ONGO"
pdb_details = pdb_details_fun(name,loading_dir)
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        if pdb_name!='4MDQ':
#            print(pdb_name, ' working')
            pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)

name = "TSG"
pdb_details = pdb_details_fun(name,loading_dir)
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)       

name = "Fusion"
pdb_details = pdb_details_fun(name,loading_dir)
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)   
#%% do the analysis for problemtric PDBs
#name = "ONGO"
#pdb_details = pdb_details_fun(name,loading_dir)
#loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
#saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
#problemtric_pdbs=["6AMB","5YFN","3GFT","5USJ","5WLB","3HHM","4OVV","5SW8","5SWG","5SWO","5SWP","5SWR","5SWT","5SX8","5SX9","5SXA","5SXB","5SXC","5SXD","5SXE","5SXF","5SXI","5SXJ","5SXK","5FI0","5IFE","4JKV","4N4W","5DIS"]
#for pdb_name in problemtric_pdbs:
#    if pdb_name!='4MDQ':
##            print(pdb_name, ' working')
#        pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)
#
#name = "TSG"
#pdb_details = pdb_details_fun(name,loading_dir)
#loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
#saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
#problemtric_pdbs=["3Q4T","3SOC","4ASX","1YPZ","2F53","5KNM","1O6S","2OMT","2OMU","2OMV","2OMX","2OMY","3HHM","4OVV","5SW8","5SWG","5SWO","5SWP","5SWR","5SWT","5SX8","5SX9","5SXA","5SXB","5SXC","5SXD","5SXE","5SXF","5SXI","5SXJ","5SXK","5GLJ"]
#for pdb_name in problemtric_pdbs:
#    pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)
#
#name = "Fusion"
#pdb_details = pdb_details_fun(name,loading_dir)
#loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
#saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
#problemtric_pdbs=["5UC4","5UCH","5UCI","5UCJ","5DIS"]
#for pdb_name in problemtric_pdbs:
#    pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)

#name = "ONGO"
#loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
#pdb_name='3QS7'
#os.chdir('/')
#os.chdir(loading_pikle_dir)
#coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
#fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
#fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
