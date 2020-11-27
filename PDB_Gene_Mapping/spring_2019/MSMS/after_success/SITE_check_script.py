# -*- coding: utf-8 -*-
"""
Created on %%(09-Sep-2018) 19.03Pm

@author: %A.Nishanth C00294860
"""
from SITE_check_class  import SITE_check
import os
threshold_length = 81
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
os.chdir('/')
os.chdir(working_dir)
files=os.listdir()

#%% 
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked"

checking_dir_PDB = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdbs_source"
for names in files:
    obj = SITE_check(working_dir,checking_dir_PDB,saving_dir,names[8:-4], threshold_length)
    obj.selecting_PDBs()
    print(names[8:-4], " Done:)")
    del obj
    #%%
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked"

checking_dir_PDB = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdbs_source"
names='uniprot-new_edited_T1.txt'
obj = SITE_check(working_dir,checking_dir_PDB,saving_dir,names[8:-4], threshold_length)
obj.selecting_PDBs()
print(names[8:-4], " Done:)")
del obj