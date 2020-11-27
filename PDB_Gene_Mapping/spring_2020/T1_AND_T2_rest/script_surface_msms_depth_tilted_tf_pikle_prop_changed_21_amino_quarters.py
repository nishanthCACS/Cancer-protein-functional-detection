# -*- coding: utf-8 -*-
"""
Created on %31-Dec-2019(1.11 P.m)

@author: %A.Nishanth C00294860
"""
import pickle
import os
from  class_surface_msms_depth_tilted_tf_pikle_prop_changed_21_amino_quarters import surface_msms_depth_MOTIF_class_quat

#%%
SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
# to load the needed PDBs
os.chdir('/')
os.chdir(SITE_dir)
SITE_satisfied_all= pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))     

  
saving_dir = 'F:/scrach_cacs_1083/optimaly_tilted_21_quarter/T_1_T_2_rest/before_thresh'
loading_pikle_dir = 'C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/2020_SITE_sat'
for pdb_name in SITE_satisfied_all:
    print(pdb_name)
    obj = surface_msms_depth_MOTIF_class_quat(loading_pikle_dir, saving_dir, pdb_name,height=128)
    del obj    
    

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
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
