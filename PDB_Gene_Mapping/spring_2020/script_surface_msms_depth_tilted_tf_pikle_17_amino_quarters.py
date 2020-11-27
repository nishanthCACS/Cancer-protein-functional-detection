# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import pickle
import os
from  class_surface_msms_depth_tilted_tf_pikle_prop_changed_21_amino_quarters import surface_msms_depth_MOTIF_class_quat

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
    
def creation_of_data(name):
#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    
    train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
    saving_dir_part = "E:/scrach_cacs_1083/optimaly_tilted_17_quarter"   
    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
    saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])
    train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
    print(name," training in progress")
    for pdb_name in train_pdb_ids_details_all:
        if pdb_name!='4MDQ':
#            print(pdb_name)
            obj = surface_msms_depth_MOTIF_class_quat(loading_pikle_dir, saving_dir_part_train, pdb_name)
            del obj
    
    saving_dir_part_test=''.join([saving_dir_part,'/Test/',name])
    test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
    print(name," testing in progress")
    
    for pdb_name in test_pdb_ids_details_all:
        if pdb_name!='4MDQ':
#            print(pdb_name)
            obj = surface_msms_depth_MOTIF_class_quat(loading_pikle_dir, saving_dir_part_test, pdb_name)
            del obj    
#%% 
name = "ONGO"
creation_of_data(name)

name = "TSG"
creation_of_data(name)
#
name = "Fusion"
creation_of_data(name)

#%%
''' One of the example has the new 21'st aminoacid'''
name = "ONGO"

#obj = surface_msms_depth_MOTIF_class_quat(loading_pikle_dir, saving_dir_part_train, pdb_name)
#del obj
##creation_of_data(name)
#loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
#    
#saving_dir_part = "G:/"   
#loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
#
#saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])
#print(name," training in progress")
#pdb_name='5wlb'
##            print(pdb_name)