# -*- coding: utf-8 -*-
"""
Created on %09-Mar-2020(9.21 P.m)

@author: %A.Nishanth C00294860

This code only retrive the PDBs already saved by 2019 -4-07-2019
spring frozen T1 and T2 PDBs
"""
import os
import pickle
from copy import deepcopy
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked')
#%%
files=os.listdir()
SITE_satisfied_files=[]
SITE_satisfied_PDBs_apart=[]
for i in range(0,len(files)):
    if files[i][-16:-2]=='SITE_satisfied':
        SITE_satisfied_files.append(files[i])
        SITE_satisfied_PDBs_apart.append(sum(pickle.load(open(files[i], "rb" )),[]))

#SITE_satisfied_PDBs=sum(SITE_satisfied_PDBs_apart,[])
#%% 
'''
Then go through each of the files and create the 21 and 17 properties  
'''
from  class_surface_msms_depth_tilted_tf_pikle_prop_changed_21_amino_quarters import surface_msms_depth_MOTIF_class_quat

#%%
def creation_of_data(name,train_pdb_ids_details_all,height):
    
    loading_pikle_dir_part='C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/after_success/'
#    saving_dir_part = 'E:/scrach_cacs_1083/Before_thresh_changed/optimaly_tilted_21_quarter/Tier_'
    saving_dir_part = 'E:/BIBM_project_data/Before_threshold_changed/17_SITE/Tier_1'
    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    saving_dir_part_train=''.join([saving_dir_part,str(name[-1])])
#    print(saving_dir_part_train)
#    print(loading_pikle_dir)
    print('')
    print("Working on ", name)
    print('')
    for pdb_name in train_pdb_ids_details_all:
        if pdb_name!='4MDQ':
#            print(pdb_name)
            obj = surface_msms_depth_MOTIF_class_quat(loading_pikle_dir, saving_dir_part_train, pdb_name,height=height)
            del obj

#for i in range(0,len(SITE_satisfied_files)):
i=3
creation_of_data(SITE_satisfied_files[i][0:-17],SITE_satisfied_PDBs_apart[i],height=200)
'''any way if quarter files mean each quatery structure is with 128 x128'''

    