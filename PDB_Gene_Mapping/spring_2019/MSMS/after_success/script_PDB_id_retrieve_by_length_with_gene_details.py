# -*- coding: utf-8 -*-
"""
Created on  %(07-April-2019) 5.19Pm

@author: %A.Nishanth C00294860
"""

from class_PDB_id_retrieve_by_length_with_gene_details import pdb_id_retrive_length_and_gene_details
from class_PDB_id_retrieve_by_length_with_gene_details import selecting_PDBs

import os
#%%
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene"
#%%
os.chdir('/')
os.chdir(working_dir)
files=os.listdir()
for names in files:
    obj = pdb_id_retrive_length_and_gene_details(working_dir,saving_dir,names[8:-4])
    obj.load_main_file()
    obj.X_ray_PDBs_with_length()
    print(names[8:-4], " Done:)")
    del obj
    
#%%
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene"
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/selected_pdbs"
for names in files:
    os.chdir('/')
    os.chdir(working_dir)
    obj = selecting_PDBs(saving_dir,names[8:-4])
    obj.selecting_PDBs()
    print(names[8:-4], " Done:)")
    del obj
    
#%% creating PDB source directory
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdbs_source"
os.chdir('/')
os.chdir(saving_dir)
for names in files:
    os.mkdir(names[8:-4])
#%% pool the PDBs together and download them
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/selected_pdbs"
os.chdir('/')
os.chdir(working_dir)
files = os.listdir()
pdbs_all=[]
for names in files:
    #select the pickle files
    if names[-1]=='p':
        tmp_pdbs = pickle.load(open(names, "rb" ) )
        pdbs_all.append(tmp_pdbs)
        
#%%
pdbs_sum= sum(pdbs_all, [])
#saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene"
#import pickle
#os.chdir('/')
#os.chdir(saving_dir)
#omitted= pickle.load(open( ''.join(["ONGO_T_2","_omitted.p"]), "rb" ) )      
#resolution= pickle.load(open( ''.join(["ONGO_T_2","_resolution.p"]), "rb" ) )      
#length= pickle.load(open( ''.join(["ONGO_T_2","_length.p"]), "rb" ) )
#Ids_info_all= pickle.load(open( ''.join(["ONGO_T_2","_Ids_info_all.p"]), "rb" ) )   
#start_end_position= pickle.load(open( ''.join(["ONGO_T_2","_start_end_position.p"]), "rb" ) )   
#unigene = pickle.load(open( ''.join(["ONGO_T_2","_uni_gene.p"]), "rb" ) )   