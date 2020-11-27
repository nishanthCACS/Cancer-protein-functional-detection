# -*- coding: utf-8 -*-
"""
Created on %(06-Nov-2018) 11.30Am

@author: %A.Nishanth C00294860
"""
import os
from class_probabilty_weigtht_initialize_gene_from_PDB_seq_length_SITE_MUST import probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST

working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"


"BLOCK 1 out of 7"
name = "ONGO"
saving_dir = "".join([working_dir, "/SITE_MUST_",name])
if not os.path.exists(saving_dir):
   os.mkdir(saving_dir)
ONGO = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO",SITE_MUST = True,site_dir=site_dir)
ONGO.method_whole_finalize()
#%% creating the model for TSG
"BLOCK 2 out of 7"
name = "TSG"
saving_dir = "".join([working_dir, "/SITE_MUST_",name])
os.mkdir(saving_dir)
TSG = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"TSG",SITE_MUST = True,site_dir=site_dir)
TSG.method_whole_finalize()

#% creating the model for Fusion
"BLOCK 3 out of 7"
name = "Fusion"
saving_dir = "".join([working_dir, "/SITE_MUST_",name])
os.mkdir(saving_dir)
Fusion = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"Fusion",SITE_MUST = True,site_dir=site_dir)
Fusion.method_whole_finalize()
#%%
"""SITE_Must_info hasn't produced yet"""
#"BLOCK 4 out of 7"
##% creating the model for TSG_Fusion
#name = "TSG_Fusion"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#TSG_Fusion = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"TSG_Fusion",SITE_MUST = True,site_dir=site_dir)
#TSG_Fusion.method_whole_finalize()
#
#"BLOCK 5 out of 7"
##% creating the model for ONGO_Fusion
#name = "ONGO_Fusion"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#ONGO_Fusion = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO_Fusion",SITE_MUST = True,site_dir=site_dir)
#ONGO_Fusion.method_whole_finalize()
#
#"BLOCK 6 out of 7"
##% creating the model for ONGO_TSG
#name = "ONGO_TSG"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#ONGO_TSG = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO_TSG",SITE_MUST = True,site_dir=site_dir)
#ONGO_TSG.method_whole_finalize()
#
#"BLOCK 7 out of 7"
##% creating the model for ONGO_TSG_Fusion
#name = "ONGO_TSG_Fusion"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#ONGO_TSG_Fusion = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO_TSG_Fusion",SITE_MUST = True,site_dir=site_dir)
#ONGO_TSG_Fusion.method_whole_finalize()
#%% Tier_2

#working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_2_pdb_pikles_length_gene"
#
#"BLOCK 1 out of 6"
#name = "ONGO"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#ONGO = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO",SITE_MUST = True,site_dir=site_dir)
#ONGO.method_whole_finalize()
#
##% creating the model for TSG
#"BLOCK 2 out of 6"
#name = "TSG"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#TSG = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"TSG",SITE_MUST = True,site_dir=site_dir)
#TSG.method_whole_finalize()
#
##% creating the model for Fusion
#"BLOCK 3 out of 6"
#name = "Fusion"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#Fusion = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"Fusion",SITE_MUST = True,site_dir=site_dir)
#Fusion.method_whole_finalize()
#
#"BLOCK 4 out of 6"
##% creating the model for ONGO_Fusion
#name = "ONGO_Fusion"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#ONGO_Fusion = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO_Fusion",SITE_MUST = True,site_dir=site_dir)
#ONGO_Fusion.method_whole_finalize()
#
#"BLOCK 5 out of 6"
##% creating the model for ONGO_TSG
#name = "ONGO_TSG"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#ONGO_TSG = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"ONGO_TSG",SITE_MUST = True,site_dir=site_dir)
#ONGO_TSG.method_whole_finalize()
#
#"BLOCK 6 out of 6"
##% creating the model for ONGO_TSG_Fusion
#name = "Notannotated"
#saving_dir = "".join([working_dir, "/SITE_MUST_",name])
#os.mkdir(saving_dir)
#Notannotated = probability_weight_initialize_gene_from_PDB_seq_length_SITE_MUST(working_dir,saving_dir,"Notannotated",SITE_MUST = True,site_dir=site_dir)
#Notannotated.method_whole_finalize()