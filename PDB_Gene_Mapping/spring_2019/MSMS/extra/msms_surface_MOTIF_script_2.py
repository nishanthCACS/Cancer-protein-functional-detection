# -*- coding: utf-8 -*-
"""
Created on %17-Feb-2019(12.10 P.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle

from msms_surface_MOTIF_class  import msms_surface_MOTIF_class

def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
#    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

#%% For Tier_1
#name = "ONGO"
#
#saving_dir = ''.join([saving_dir_part,name])
#pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#
#pdb_details = pdb_details_fun(name,loading_dir)
#%%print("considering: ",pdb_details[9][0])
#pdb_name = pdb_details[9][0]
#pdb_file = ''.join([pdb_name,".pdb"])
#               
#msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#msms_obj.retrieve_info()
#del msms_obj
#for i in range(0,len(pdb_details)):
#    pdb_list = pdb_details[i]
#    for pdb_name in pdb_list:
#        pdb_file = ''.join([pdb_name,".pdb"])
#                       
#        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#        msms_obj.retrieve_info()
#        del msms_obj

#%%
name = "TSG"

saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
pdb_details = pdb_details_fun(name,loading_dir)
print(name)
for pdb_list in pdb_details:
    for pdb_name in pdb_list:
        pdb_file = ''.join([pdb_name,".pdb"])
                       
        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
        msms_obj.retrieve_info()
        del msms_obj
#    
#name = "Fusion"
#
#saving_dir = ''.join([saving_dir_part,name])
#pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#pdb_details = pdb_details_fun(name,loading_dir)
#
#for pdb_list in pdb_details:
#    for pdb_name in pdb_list:
#        pdb_file = ''.join([pdb_name,".pdb"])
#                       
#        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#        msms_obj.retrieve_info()
#        del msms_obj