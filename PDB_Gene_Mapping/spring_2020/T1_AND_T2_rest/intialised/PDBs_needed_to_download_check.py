# -*- coding: utf-8 -*-
"""
Created on %(26-Apr-2020) 2.05Pm

@author: %A.Nishanth C00294860

This function check the downloaded PDBs is enough or Do I need to download it again
"""
import os
import pickle
from copy import deepcopy
working_dir_part="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"

#%% PDBs needed
PDBs_needed=[]
for Tier in [1,2]:
    names_whole=os.listdir(''.join([working_dir_part,'pdb_details_length_source_2020_V_91/T_',str(Tier)]))
    loading_dir=''.join([working_dir_part,"Tier_",str(Tier),"_pdb_pikles_length_gene"])
    os.chdir('/')
    os.chdir(loading_dir)
    for n in names_whole:
        name=n[4:-4]
        Pickles_t= pickle.load(open(''.join([name,"_PDBs.p"]), "rb"))        
        PDBs_needed.append(deepcopy(Pickles_t))
PDBS_all=sum(PDBs_needed,[])
PDBS_all = list(set(PDBS_all)) 
#%%
'''
First remove the PDBs already analised 2018 to current
checked with the 2019one
To get the PDBs no need to go through SITE checking
ANd those needed to go through SITE checking 
Since the properties with SITE give higher preformance than without SITE
'''
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked')
pikle_list_2019=os.listdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked')
pikle_list_sel=[]
PDBS_all_checked=[]
for n in pikle_list_2019:
#        name=n[4:-4]
    if n[-29:-2] =='gene_list_thres_sat_PDB_ids':
        pikle_list_sel.append(n)
        Pickles_t= pickle.load(open(n, "rb"))    
        PDBS_all_checked.append(deepcopy(Pickles_t))
PDBS_all_checked=sum(PDBS_all_checked,[])
PDBS_all_checked=sum(PDBS_all_checked,[])

PDBS_all_checked = list(set(PDBS_all_checked)) 
#%%
from Class_PDB_id_retrieve_by_length_spring_V_91  import selecting_PDBs

saving_dir='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene'
pikle_list_2018_T_1=os.listdir(saving_dir)
os.chdir(saving_dir)
#Fusion_T_2_gene_list_thres_sat_PDB_ids
PDBS_all_checked_T1=[]
for n in pikle_list_2018_T_1:
#        name=n[4:-4]
    if len(n)>14:
        if n[-14:-2] =='Ids_info_all':
#            Pickles_t= pickle.load(open(n, "rb"))    
            name=n[0:-15]
            if name=='ONGO' or name=='TSG' or name=='Fusion':
                print("Working ", name)
                obj=selecting_PDBs(saving_dir,name)
                Pickles_t=obj.return_only_selecting_PDBs()
                PDBS_all_checked_T1.append(deepcopy(Pickles_t))
                print("done ", name," :)")
PDBS_all_checked_T1=sum(PDBS_all_checked_T1,[])
#PDBS_all_checked_T1=sum(PDBS_all_checked_T1,[])
PDBS_all_checked_T1 = list(set(PDBS_all_checked_T1)) 
#%%
PDBS_all_checked_To_gether=PDBS_all_checked+PDBS_all_checked_T1
PDBS_all_checked_To_gether = list(set(PDBS_all_checked_To_gether)) 
#            PDBS_all_checked.append(deepcopy(Pickles_t))
#%%
needed_chk=[]
for pdb in PDBS_all:
    if pdb not in PDBS_all_checked_To_gether:
        needed_chk.append(pdb)
#%% download 
        
#%%        
'''
To findout the howmany PDBs exist in the 2020 data that satisfy 81 threshold
'''

Tier=1
PDBs_needed=[]
names_whole=os.listdir(''.join([working_dir_part,'pdb_details_length_source_2020_V_91/T_',str(Tier)]))
loading_dir=''.join([working_dir_part,"Tier_",str(Tier),"_pdb_pikles_length_gene"])

os.chdir('/')
os.chdir(loading_dir)
for n in names_whole:
    name=n[4:-4]
    if name=="ONGO" or name=="TSG" or name=="Fusion":
        Pickles_t= pickle.load(open(''.join([name,"_PDBs.p"]), "rb"))        
        PDBs_needed.append(deepcopy(Pickles_t))
PDBS_all=sum(PDBs_needed,[])
PDBS_all = list(set(PDBS_all)) 