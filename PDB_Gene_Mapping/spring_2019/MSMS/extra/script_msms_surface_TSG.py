# -*- coding: utf-8 -*-
"""
Created on %25-Feb-2019(12.00 P.m)

@author: %A.Nishanth C00294860

This script is made after SITE pdbs run by MSMS tool
"""
import os
import pickle

from class_msms_surface  import msms_surface_class
#%%
def pdb_details_fun(name):

    loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    os.chdir("/")
    os.chdir(loading_dir)
    #    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    site_sat_pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    if name=='Fusion':
        print('f')
#        clean_unigene =pickle.load(open('Fusion_thres_sat_gene_list.p', "rb" ))
        clean_pdbs =pickle.load(open('Fusion_gene_list_thres_sat_PDB_ids.p', "rb" ))
    else:
        print('O or T')
#        clean_unigene =pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ))
        clean_pdbs =pickle.load(open( ''.join([name,"_clean_O_T_pdbs.p"]), "rb" ))
    
    clean_pdbs_all=sum(clean_pdbs,[])
    site_sat_pdb_details_all=sum(site_sat_pdb_details,[])
    needed_pdbs=[]
    for pdb in clean_pdbs_all:
        if pdb not in site_sat_pdb_details_all:
            needed_pdbs.append(pdb)
            
    return needed_pdbs

pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

#%% For Tier_1
name = "TSG"
saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#pdb_name='721p'
pdb_details = pdb_details_fun(name)
for i in range(0,len(pdb_details)):
    pdb_name = pdb_details[i]
    pdb_file = ''.join([pdb_name,".pdb"])
    msms_obj = msms_surface_class(pdb_source_dir, saving_dir, pdb_name)
    msms_obj.retrieve_info()
    del msms_obj
   
