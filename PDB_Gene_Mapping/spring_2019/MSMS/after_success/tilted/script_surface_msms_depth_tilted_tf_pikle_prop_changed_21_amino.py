# -*- coding: utf-8 -*-
"""
Created on %19-Nov-2019(10.33 A.m)

@author: %A.Nishanth C00294860
"""
import pickle
import os
from  class_surface_msms_depth_tilted_tf_pikle_prop_changed_21_amino import surface_msms_depth_MOTIF_class
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
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/unigene/optimaly_tilted"   
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/neg_size_19_21_aminoacids"
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/neg_size_20_21_aminoacids"
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/neg_size_15_21_aminoacids"

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
            obj = surface_msms_depth_MOTIF_class(loading_pikle_dir, saving_dir_part_train, pdb_name,height=height)
            del obj
    
    saving_dir_part_test=''.join([saving_dir_part,'/Test/',name])
    test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
    print(name," testing in progress")
    
    for pdb_name in test_pdb_ids_details_all:
        if pdb_name!='4MDQ':
#            print(pdb_name)
            obj = surface_msms_depth_MOTIF_class(loading_pikle_dir, saving_dir_part_test, pdb_name,height=height)
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
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
