# -*- coding: utf-8 -*-
"""
Created on %(02-Sep-2018) 19.40Pm

@author: %A.Nishanth C00294860
"""
import os
from class_probabilty_weigtht_initialize_gene_from_PDB_seq_length import probability_weight_initialize_gene_from_PDB_seq_length

working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"

"BLOCK 1 out of 7"
name = "ONGO"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
ONGO = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO")
ONGO.method_whole_finalize()

#%% creating the model for TSG
"BLOCK 2 out of 7"
name = "TSG"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
TSG = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"TSG")
TSG.method_whole_finalize()

#% creating the model for Fusion
"BLOCK 3 out of 7"
name = "Fusion"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
Fusion = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"Fusion")
Fusion.method_whole_finalize()

"BLOCK 4 out of 7"
#% creating the model for TSG_Fusion
name = "TSG_Fusion"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
TSG_Fusion = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"TSG_Fusion")
TSG_Fusion.method_whole_finalize()

"BLOCK 5 out of 7"
#% creating the model for ONGO_Fusion
name = "ONGO_Fusion"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
ONGO_Fusion = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO_Fusion")
ONGO_Fusion.method_whole_finalize()

"BLOCK 6 out of 7"
#% creating the model for ONGO_TSG
name = "ONGO_TSG"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
ONGO_TSG = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO_TSG")
ONGO_TSG.method_whole_finalize()

"BLOCK 7 out of 7"
#% creating the model for ONGO_TSG_Fusion
name = "ONGO_TSG_Fusion"
saving_dir = "".join([working_dir, "/",name])
#os.mkdir(saving_dir)
ONGO_TSG_Fusion = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO_TSG_Fusion")
ONGO_TSG_Fusion.method_whole_finalize()
#%% Tier_2
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_2_pdb_pikles_length_gene"

"BLOCK 1 out of 6"
name = "ONGO"
saving_dir = "".join([working_dir, "/",name])
os.mkdir(saving_dir)
ONGO = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO")
ONGO.method_whole_finalize()

#% creating the model for TSG
"BLOCK 2 out of 6"
name = "TSG"
saving_dir = "".join([working_dir, "/",name])
os.mkdir(saving_dir)
TSG = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"TSG")
TSG.method_whole_finalize()

#% creating the model for Fusion
"BLOCK 3 out of 6"
name = "Fusion"
saving_dir = "".join([working_dir, "/",name])
os.mkdir(saving_dir)
Fusion = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"Fusion")
Fusion.method_whole_finalize()

"BLOCK 4 out of 6"
#% creating the model for ONGO_Fusion
name = "ONGO_Fusion"
saving_dir = "".join([working_dir, "/",name])
os.mkdir(saving_dir)
ONGO_Fusion = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO_Fusion")
ONGO_Fusion.method_whole_finalize()

"BLOCK 5 out of 6"
#% creating the model for ONGO_TSG
name = "ONGO_TSG"
saving_dir = "".join([working_dir, "/",name])
os.mkdir(saving_dir)
ONGO_TSG = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"ONGO_TSG")
ONGO_TSG.method_whole_finalize()

"BLOCK 6 out of 6"
#% creating the model for ONGO_TSG_Fusion
name = "Notannotated"
saving_dir = "".join([working_dir, "/",name])
os.mkdir(saving_dir)
Notannotated = probability_weight_initialize_gene_from_PDB_seq_length(working_dir,saving_dir,"Notannotated")
Notannotated.method_whole_finalize()