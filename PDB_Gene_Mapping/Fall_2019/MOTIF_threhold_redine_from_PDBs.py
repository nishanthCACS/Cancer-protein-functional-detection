# -*- coding: utf-8 -*-
"""
Created on %31-Dec-2019(11.39 A.m)

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import csv
from copy import deepcopy
import pickle

def pdbs_no_software_must_info(name):
    loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"

    os.chdir("/")
    os.chdir(loading_dir)
    pdbs_not_software_must = pickle.load(open(''.join([name,"_pdbs_not_software_must.p"]), "rb" ))
#    pdbs_not_software_must_exp_type =pickle.load(open(''.join([name,"_pdbs_not_software_must_exp_type.p"]),"rb"))
#    pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_must_gene.p"]), "rb")) 

    pdb_name=pdbs_not_software_must[0]
    
    '''Since only the 3ZRC satisfied just remove the details of MOTIF about it'''
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    
    saving_dir = ''.join([saving_dir_part,name])
    os.chdir('/')
    os.chdir(saving_dir)
    MOTIF_indexs_all = pickle.load(open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ) ) 
    coordinates = pickle.load(open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ) ) 
    amino_acid = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ) ) 
    fin_res_depth_all = pickle.load(open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ) ) 
    fin_ca_depth_all = pickle.load(open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ) ) 
    
    return MOTIF_indexs_all,coordinates,amino_acid,fin_res_depth_all,fin_ca_depth_all


name = "TSG"
MOTIF_indexs_all,coordinates,amino_acid,fin_res_depth_all,fin_ca_depth_all=pdbs_no_software_must_info(name)
#%% get the depths of the Ca depth and res_depth of MOTIFs
MOTIF_indexs_all_flatten = sum(MOTIF_indexs_all,[])
depths_ca_MOTIF=[]
depths_res_MOTIF=[]
for i in MOTIF_indexs_all_flatten:
    depths_ca_MOTIF.append(fin_ca_depth_all[i])
    depths_res_MOTIF.append(fin_res_depth_all[i])
print('depths_ca_MOTIF maximum: ',max(depths_ca_MOTIF))  
print('fin_ca_depth_all maximum: ',max(fin_ca_depth_all))  
print('')
print('depths_res_MOTIF maximum: ',max(depths_res_MOTIF))  
print('fin_res_depth_all maximum: ',max(fin_res_depth_all)) 

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
 
os.chdir("/")
os.chdir(loading_dir)
pickle.dump(max(depths_ca_MOTIF), open("max_depths_ca_MOTIF.p", "wb"))  
pickle.dump(max(depths_res_MOTIF), open("max_depths_res_MOTIF.p", "wb"))  
#correspond PDBs depth
pickle.dump(max(fin_res_depth_all), open("Author_3ZRC_max_fin_res_depth_all.p", "wb"))  
pickle.dump(max(fin_ca_depth_all), open("Author_3ZRC_max_fin_ca_depth_all.p", "wb"))  
#%%
'''Retrieve the 3ZRC info'''
#import os
#import pickle
#
#from class_msms_surface_21_amino  import msms_surface_MOTIF_class
#'''Here only fixing the PDBs with 21st aminoacid'''
#
#
#def retrieve_info_PDB(pdb_name,name):
#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
#
#    saving_dir = ''.join([saving_dir_part,name])
#    pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#    msms_obj.retrieve_info()
#    del msms_obj
#name = "TSG"
#retrieve_info_PDB(pdb_name,name)  
#print(pdb_name)
import os
import pickle

from class_msms_surface_21_amino  import msms_surface_MOTIF_class
'''Here only fixing the PDBs with 21st aminoacid'''


def retrieve_info_PDB(pdb_name,name):
    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

    saving_dir = ''.join([saving_dir_part,name])
    pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
    msms_obj.retrieve_info()
    del msms_obj
name='ONGO'
pdb_name = '6AXG'
retrieve_info_PDB(pdb_name,name)  
print(pdb_name, " done :)")