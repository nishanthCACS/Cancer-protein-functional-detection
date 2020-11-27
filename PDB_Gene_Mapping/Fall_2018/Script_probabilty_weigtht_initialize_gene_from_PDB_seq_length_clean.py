# -*- coding: utf-8 -*-
"""
Created on %15-November-2018(21.02pm)

@author: %A.Nishanth C00294860
"""


import os
from class_probabilty_weigtht_initialize_gene_from_PDB_seq_length_clean import probability_weight_initialize_gene_from_PDB_seq_length_clean

clean_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
SITE_MUST = False
clean = True

"BLOCK 1 out of 7"
name = "ONGO"

if clean and SITE_MUST:
    saving_dir = "".join([working_dir, "/clean_SITE_MUST_",name])
elif clean:
    saving_dir = "".join([working_dir, "/clean_",name])
    print("clean_only!")
if not os.path.exists(saving_dir):
   os.mkdir(saving_dir)

ONGO = probability_weight_initialize_gene_from_PDB_seq_length_clean(working_dir,saving_dir,"ONGO",clean=clean,clean_dir=clean_dir,SITE_MUST=SITE_MUST,site_dir=site_dir)
ONGO.method_whole_finalize()
#%% creating the model for TSG
"BLOCK 2 out of 7"
name = "TSG"
if clean and SITE_MUST:
    saving_dir = "".join([working_dir, "/clean_SITE_MUST_",name])
elif clean:
    saving_dir = "".join([working_dir, "/clean_",name])
if not os.path.exists(saving_dir):
   os.mkdir(saving_dir)
TSG = probability_weight_initialize_gene_from_PDB_seq_length_clean(working_dir,saving_dir,"TSG",clean=clean,clean_dir=clean_dir,SITE_MUST=SITE_MUST,site_dir=site_dir)
TSG.method_whole_finalize()

#% creating the model for Fusion
"BLOCK 3 out of 7"
name = "Fusion"
if clean and SITE_MUST:
    saving_dir = "".join([working_dir, "/clean_SITE_MUST_",name])
elif clean:
    saving_dir = "".join([working_dir, "/clean_",name])
if not os.path.exists(saving_dir):
   os.mkdir(saving_dir)
Fusion = probability_weight_initialize_gene_from_PDB_seq_length_clean(working_dir,saving_dir,"Fusion",clean=clean,clean_dir=clean_dir,SITE_MUST=SITE_MUST,site_dir=site_dir)
Fusion.method_whole_finalize()
#%%
