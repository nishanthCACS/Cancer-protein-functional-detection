# -*- coding: utf-8 -*-
"""
Created on %06-Dec-2019 at 3.38pm

@author: %A.Nishanth C00294860
"""
from class_retrieve_PDB_issue  import msms_surface_class


pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
#% For Tier_1
#name = "ONGO"

#pdb_name = '5C3I'
#saving_dir = ''.join([saving_dir_part,name])
#pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#msms_obj = msms_surface_class(pdb_source_dir, saving_dir, pdb_name)
#coordinates,amino_acid,info_all_cordinates,atom_number_list_c,info_all_cordinates,atom_number_list_c = msms_obj.retrieve_info()
#del msms_obj
name = "TSG"

pdb_name = '4WYQ'
saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
msms_obj = msms_surface_class(pdb_source_dir, saving_dir, pdb_name)
coordinates,amino_acid,info_all_cordinates,atom_number_list_c,info_all_cordinates,atom_number_list_c = msms_obj.retrieve_info()
del msms_obj
#%%
   
