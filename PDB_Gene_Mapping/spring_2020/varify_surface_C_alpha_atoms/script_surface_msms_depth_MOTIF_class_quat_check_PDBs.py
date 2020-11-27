# -*- coding: utf-8 -*-
"""
Created on %31-Mar-2020(3.31 P.m)

@author: %A.Nishanth C00294860
"""
import pickle
import os
from  class_surface_msms_depth_MOTIF_class_quat_check_PDBs import surface_msms_depth_MOTIF_class_quat_check_PDBs

#%%
def pdb_details_fun(name):
#    name='ONGO'    
    loading_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/suceed_unigene"   
  
    os.chdir("/")
    os.chdir(loading_dir)
    test_pdb_ids_details  = pickle.load(open( ''.join([name,"_test_pdb_ids_details.p"]), "rb" ) )
    train_pdb_ids_details =  pickle.load(open( ''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
    test_unigene_details =  pickle.load(open( ''.join([name,"_test_uni_gene.p"]), "rb" )) 
    return train_pdb_ids_details,test_pdb_ids_details,test_unigene_details
    
def creation_of_data(name,height):
#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    
    train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
#    saving_dir_part = 'C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2020/old_thresh_optimal_tilted_SITE_MUST_21_amino'
#    saving_dir_part = 'E:/scrach_cacs_1083/Before_thresh_changed/optimaly_tilted_21_quarter'
    saving_dir_part = 'E:/scrach_cacs_1083/optimaly_tilted_21_quarter'

    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
    saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])
    train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
    print(name," training in progress")
    
#    pdb_chk_list=['1W72','2AK4','2H26','4PJE','5C08','5C0A','5C0B','5C0C','5EUO','5U6Q']
#    for pdb_name in pdb_chk_list:
#        print(pdb_name)
#        obj = surface_msms_depth_MOTIF_class(loading_pikle_dir, saving_dir_part_train, pdb_name,height=height)
#        del obj
    
    for pdb_name in train_pdb_ids_details_all:
        if pdb_name!='4MDQ':
#            print(pdb_name)
            obj = surface_msms_depth_MOTIF_class_quat_check_PDBs(loading_pikle_dir, saving_dir_part_train, pdb_name,height=height)
            del obj
    
    saving_dir_part_test=''.join([saving_dir_part,'/Test/',name])
    test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
    print(name," testing in progress")
    
    for pdb_name in test_pdb_ids_details_all:
        if pdb_name!='4MDQ':
#            print(pdb_name)
            obj = surface_msms_depth_MOTIF_class_quat_check_PDBs(loading_pikle_dir, saving_dir_part_test, pdb_name,height=height)
            del obj    
#%%
height=200
name = "ONGO"
creation_of_data(name,height)
#%
name = "TSG"
creation_of_data(name,height)
#
name = "Fusion"
creation_of_data(name,height)

#%%
#import numpy as np
#name = "ONGO"
#
#saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/21_aminoacids_SITE_MUST"
#saving_dir_part_train=''.join([saving_dir_part,'/Test/',name])
#os.chdir('/')
#os.chdir(saving_dir_part_train)
#prop = np.load('121p.npy')

#%%
name = "ONGO"

loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
pdb_name='721P'
os.chdir('/')
os.chdir(loading_pikle_dir)
coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))         

#%%
clean_must_sat_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Before_thresh/20_amino_acids/17-prop"

os.chdir('/')
os.chdir(clean_must_sat_dir)
site_sat=  pickle.load(open( ''.join(['OG',"_train_PDBs.p"]), "rb"))
if '721P.npy' in site_sat:
    print("fdysdf")