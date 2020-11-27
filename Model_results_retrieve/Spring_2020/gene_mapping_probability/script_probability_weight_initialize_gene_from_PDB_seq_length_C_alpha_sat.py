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
#for organise the results or performance
clean_must_sat_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Before_thresh/20_amino_acids/17-prop"
threshold_dir= "old_thresh/20_amino_acids/Clean_no_overlap_at_all_20_amino_prop_with_SITE/"
    
#%%
"BLOCK 1 out of 7"
name = "ONGO"

saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
if not os.path.exists(saving_dir):
   os.mkdir(saving_dir)
ONGO = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,sat_PDB_dir=clean_must_sat_dir)
ONGO.method_whole_finalize()
#%% creating the model for TSG
"BLOCK 2 out of 7"
name = "TSG"
saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
TSG = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,sat_PDB_dir=clean_must_sat_dir)
TSG.method_whole_finalize()

#% creating the model for Fusion
"BLOCK 3 out of 7"
name = "Fusion"
saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
Fusion = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,sat_PDB_dir=clean_must_sat_dir)
Fusion.method_whole_finalize()