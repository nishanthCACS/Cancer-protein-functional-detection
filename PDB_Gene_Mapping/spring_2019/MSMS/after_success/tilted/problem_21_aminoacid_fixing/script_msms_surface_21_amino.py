# -*- coding: utf-8 -*-
"""
Created on %25-Feb-2019(12.00 P.m)

@author: %A.Nishanth C00294860

This script is made after SITE pdbs run by MSMS tool
"""
import os
import pickle

from class_msms_surface_21_amino  import msms_surface_MOTIF_class
'''Here only fixing the PDBs with 21st aminoacid'''


def retrieve_info_PDB(pdb_name,name):
    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

    saving_dir = ''.join([saving_dir_part,name])
    pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
    msms_obj.retrieve_info()
    del msms_obj

#%% For Tier_1
name = "ONGO"
#retrieve_info_PDB('5WLB',name)   
#retrieve_info_PDB('3GT8',name)   
#retrieve_info_PDB('3LZB',name)   
name = "TSG"
retrieve_info_PDB('2H26',name)  