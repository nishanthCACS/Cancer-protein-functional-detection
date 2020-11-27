# -*- coding: utf-8 -*-
"""
Created on %17-Feb-2019(12.10 P.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle

from class_msms_surface  import msms_surface_MOTIF_class
#%%

def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked"
pdb_source_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdbs_source"

saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/after_success/"

name = 'TSG_T_2'
saving_dir = ''.join([saving_dir_part,name])
os.mkdir(saving_dir)
print(name, " started")
pdb_details = pdb_details_fun(name,loading_dir)
for pdb_list in pdb_details:
    for pdb_name in pdb_list:
        pdb_file = ''.join([pdb_name,".pdb"])
                       
        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
        msms_obj.retrieve_info()
        del msms_obj
print(name, " done")
