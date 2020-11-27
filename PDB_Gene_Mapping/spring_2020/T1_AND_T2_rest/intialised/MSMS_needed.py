# -*- coding: utf-8 -*-
"""
Created on %27-Apr-2020(at 6.20pm)

@author: %A.Nishanth C00294860
"""
import os
import pickle
from class_msms_surface_21_amino  import msms_surface_MOTIF_class

SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
# to load the needed PDBs
os.chdir('/')
os.chdir(SITE_dir)
SITE_satisfied_all= pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))     
#%%
'''
Earlier 21 amino acid fixing is done in 
path_1:
'''
#path_1='C:\Users\nishy\Documents\Projects_UL\Continues BIBM\a----------PDB_Gene_Mapping\spring_2019\MSMS\after_success\tilted\problem_21_aminoacid_fixing'
pdb_source_dir='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/2020_PDBs_needed/pooled'
MSMS_saving_dir='C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/2020_SITE_sat'
i=0
for pdb_name in SITE_satisfied_all:
    print("Running on i: ",i)
    print("Remaining PDBs: ",len(SITE_satisfied_all)-i+1)
    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, MSMS_saving_dir, pdb_name)
    msms_obj.retrieve_info()
    i=i+1
    
#%%
'''
Go through the created files and Findout if any PDB left

If no PDB is print then everytrhing is perfect
'''

import os
import pickle
MSMS_saving_dir='C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/2020_SITE_sat'
MSMS_files=os.listdir(MSMS_saving_dir)
pdb_msms=[]
for files in MSMS_files:
    if files[0:10]=='amino_acid':
        pdb_msms.append(files[-6:-2])
SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
# to load the needed PDBs
os.chdir('/')
os.chdir(SITE_dir)
SITE_satisfied_all= pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))     
for pdbs in SITE_satisfied_all:
    if pdbs not in pdb_msms:
        print(pdbs)
        
        #%%
SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
os.chdir('/')
os.chdir(SITE_dir)
SITE_satisfied_temp= pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))  
# since both 21 amino acids and 20 amino acids not-satisfied in have same PDBs
not_satisfied_model=pickle.load(open('not_satisfied_model.p', "rb"))  #the PDB data unable to convert to supported quarter model formation 