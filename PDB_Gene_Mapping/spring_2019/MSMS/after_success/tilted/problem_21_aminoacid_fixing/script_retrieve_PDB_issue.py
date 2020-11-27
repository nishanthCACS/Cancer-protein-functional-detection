# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""


from class_retrieve_PDB_issue  import msms_surface_MOTIF_class
#%%


pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
#%% For Tier_1
#name = "ONGO"
name = "TSG"
saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#pdb_name='721p'

pdb_name = '2H26'
pdb_file = ''.join([pdb_name,".pdb"])
msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
MOTIF_indexs_all,coordinates,amino_acid,info_all_cordinates,atom_number_list_c = msms_obj.retrieve_info()
del msms_obj

   
