# -*- coding: utf-8 -*-
"""
Created on %(06-Nov-2018) 11.30Am

@author: %A.Nishanth C00294860

small change to fit with all data PDBs in primary sequence witch has atleast one C_alpha atom for surface in the given property conditions
"""
import os
from class_probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat import probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat

working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
'''To make the weightage 21 properties(which include SITE)
First clean must no overlapped PDBs at all
'''
def call_PDB_length_retriever(working_dir,threshold_dir,clean_must_sat_dir):
    "BLOCK 1 out of 7"
    name = "ONGO"
    
    saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
    os.chdir('/')
    if not os.path.exists(saving_dir):
       os.makedirs(saving_dir)
    ONGO = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,sat_PDB_dir=clean_must_sat_dir)
    ONGO.method_whole_finalize()
    #% creating the model for TSG
    "BLOCK 2 out of 7"
    name = "TSG"
    os.chdir('/')
    saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    TSG = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,sat_PDB_dir=clean_must_sat_dir)
    TSG.method_whole_finalize()
    
    #% creating the model for Fusion
    "BLOCK 3 out of 7"
    name = "Fusion"
    os.chdir('/')
    saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    Fusion = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,sat_PDB_dir=clean_must_sat_dir)
    Fusion.method_whole_finalize()
#for organise the results or performance
#%%
clean_must_sat_dir_part_0 = "G:/ULL_Project/optimally_tilted/quater_models/"
clean_must_sat_dir_part_save_parts=['Fully_clean/SITE/','Fully_clean/W_OUT_SITE/']
for part in [0,1]:
    clean_must_sat_dir_part=''.join([clean_must_sat_dir_part_0,clean_must_sat_dir_part_save_parts[part]])
    os.chdir('/')
    os.chdir(clean_must_sat_dir_part)
    
    c_alpha_surface_cond= os.listdir()
    for clean_dir in range(0,len(c_alpha_surface_cond)):
        clean_must_sat_dir_part_20=''.join([c_alpha_surface_cond[clean_dir],'/20_amino_acid/'])
        clean_must_sat_dir=''.join([clean_must_sat_dir_part,clean_must_sat_dir_part_20,'K_fold'])
        threshold_dir= ''.join([clean_must_sat_dir_part_save_parts[part],clean_must_sat_dir_part_20])
        print("Working on: " ,clean_must_sat_dir_part_save_parts[part],clean_must_sat_dir_part_20)
        call_PDB_length_retriever(working_dir,threshold_dir,clean_must_sat_dir)
    
    
        clean_must_sat_dir_part_21=''.join([c_alpha_surface_cond[clean_dir],'/21_amino_acid/'])
        clean_must_sat_dir=''.join([clean_must_sat_dir_part,clean_must_sat_dir_part_21,'K_fold'])
        threshold_dir= ''.join([clean_must_sat_dir_part_save_parts[part],clean_must_sat_dir_part_21])
        print("Working on: " ,clean_must_sat_dir_part_save_parts[part],clean_must_sat_dir_part_21)
        call_PDB_length_retriever(working_dir,threshold_dir,clean_must_sat_dir)