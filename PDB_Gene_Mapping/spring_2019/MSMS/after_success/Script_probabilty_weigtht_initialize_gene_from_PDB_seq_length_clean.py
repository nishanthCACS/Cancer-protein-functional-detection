# -*- coding: utf-8 -*-
"""
Created on %8-April-2019(04.22Pm)

@author: %A.Nishanth C00294860
"""
import os
from class_probabilty_weigtht_initialize_gene_from_PDB_seq_length_clean import probability_weight_initialize_gene_from_PDB_seq_length_clean
#%% preintialise
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/"
clean_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/"
site_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked"

SITE_MUST = True
clean = False# since this is only testing

# to retrieve the names
naming_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
os.chdir('/')
os.chdir(naming_dir)
files=os.listdir()
#%%
for names in files:
    name = names[8:-4]
    print("Going to work on: ",name)

    saving_dir = "".join([working_dir,name])
    if not os.path.exists(saving_dir):
       os.mkdir(saving_dir)
    obj = probability_weight_initialize_gene_from_PDB_seq_length_clean(working_dir,saving_dir,name,clean=clean,clean_dir=clean_dir,SITE_MUST=SITE_MUST,site_dir=site_dir)
    obj.method_whole_finalize()
    del obj
    print("Done: ",name," ---------------vin :)") 
#%%
name = files[1][8:-4]
saving_dir = "".join([working_dir,name])
if not os.path.exists(saving_dir):
   os.mkdir(saving_dir)
obj = probability_weight_initialize_gene_from_PDB_seq_length_clean(working_dir,saving_dir,name,clean=clean,clean_dir=clean_dir,SITE_MUST=SITE_MUST,site_dir=site_dir)
obj.method_whole_finalize()
del obj    