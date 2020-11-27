# -*- coding: utf-8 -*-
"""
Created on  %(26-Aug-2018) 2.38Pm

@author: %A.Nishanth C00294860
"""

from class_PDB_id_retrieve_by_length_with_gene_details import pdb_id_retrive_length_and_gene_details

""" Tier_1"""
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_details_length_source"
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
#%% creating the model for ONGO
"BLOCK 1 out of 7"
ONGO = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO")
ONGO.load_main_file()
ONGO.X_ray_PDBs_with_length()
#% creating the model for TSG
"BLOCK 2 out of 7"
TSG = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"TSG")
TSG.load_main_file()
TSG.X_ray_PDBs_with_length()
#% creating the model for Fusion
"BLOCK 3 out of 7"
Fusion = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"Fusion")
Fusion.load_main_file()
Fusion.X_ray_PDBs_with_length()
"BLOCK 4 out of 7"
#% creating the model for TSG_Fusion
TSG_Fusion = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"TSG_Fusion")
TSG_Fusion.load_main_file()
TSG_Fusion.X_ray_PDBs_with_length()
"BLOCK 5 out of 7"
#% creating the model for ONGO_Fusion
ONGO_Fusion = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO_Fusion")
ONGO_Fusion.load_main_file()
ONGO_Fusion.X_ray_PDBs_with_length()
"BLOCK 6 out of 7"
#% creating the model for ONGO_TSG
ONGO_TSG = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO_TSG")
ONGO_TSG.load_main_file()
ONGO_TSG.X_ray_PDBs_with_length()
"BLOCK 7 out of 7"
#% creating the model for ONGO_TSG_Fusion
ONGO_TSG_Fusion = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO_TSG_Fusion")
ONGO_TSG_Fusion.load_main_file()
ONGO_TSG_Fusion.X_ray_PDBs_with_length()
#%%
""" Tier_2"""
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_2_pdb_details_length_source"
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_2_pdb_pikles_length_gene"
#%% creating the model for ONGO
"BLOCK 1 out of 6"
ONGO = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO")
ONGO.load_main_file()
ONGO.X_ray_PDBs_with_length()
#% creating the model for TSG
"BLOCK 2 out of 6"
TSG = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"TSG")
TSG.load_main_file()
TSG.X_ray_PDBs_with_length()
#% creating the model for Fusion
"BLOCK 3 out of 6"
Fusion = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"Fusion")
Fusion.load_main_file()
Fusion.X_ray_PDBs_with_length()
"BLOCK 4 out of 6"
#% creating the model for ONGO_Fusion
ONGO_Fusion = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO_Fusion")
ONGO_Fusion.load_main_file()
ONGO_Fusion.X_ray_PDBs_with_length()
"BLOCK 5 out of 6"
#% creating the model for ONGO_TSG
ONGO_TSG = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"ONGO_TSG")
ONGO_TSG.load_main_file()
ONGO_TSG.X_ray_PDBs_with_length()
"BLOCK 6 out of 7"
#% creating the model for Notannotated
Notannotated = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,"Notannotated")
Notannotated.load_main_file()
Notannotated.X_ray_PDBs_with_length()
