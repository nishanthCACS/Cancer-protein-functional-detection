# -*- coding: utf-8 -*-
"""
Created on %(26-Apr-2020) 1.35Pm

@author: %A.Nishanth C00294860
"""
import os
from Class_PDB_id_retrieve_by_length_spring_V_91  import PDB_id_retrieve_by_length_spring_V_91
    
working_dir_part="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"
for Tier in [1,2]:
    names_whole=os.listdir(''.join([working_dir_part,'pdb_details_length_source_2020_V_91/T_',str(Tier)]))
    for n in names_whole:
        name=n[4:-4]
        print("Working ",Tier," ", name)
        PDB_id_retrieve_by_length_spring_V_91(name,Tier,working_dir_part)
        print("done ",Tier," ", name," :)")
#%%
from Class_PDB_id_retrieve_by_length_spring_V_91  import selecting_PDBs
for Tier in [1,2]:
    names_whole=os.listdir(''.join([working_dir_part,'pdb_details_length_source_2020_V_91/T_',str(Tier)]))
    saving_dir=''.join([working_dir_part,"Tier_",str(Tier),"_pdb_pikles_length_gene"])
    for n in names_whole:
        name=n[4:-4] 
        print("Working ",Tier," ", name)
        obj=selecting_PDBs(saving_dir,name)
        obj.selecting_PDBs()
        print("done ",Tier," ", name," :)")