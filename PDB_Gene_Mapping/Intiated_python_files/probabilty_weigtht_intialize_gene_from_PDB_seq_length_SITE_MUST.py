# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
import os
import pickle

name="ONGO"
os.chdir('/')
os.chdir(site_dir)

site_sat = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]),  "rb"))      
site_sat_flatten = sum(site_sat, [])    
#%%
threshold_length = 81
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
 # first load the files needed
os.chdir('/')
os.chdir(working_dir)

"""
This fumction take the 
name: ONGO or TSG or Fusion
threshold_length: PDBid has atleast this length in gene to be selected 

Ids_info_all: THIS ONLY CONTAIIN X-RAY DATA
"""
Ids_info_all = pickle.load(open( ''.join([name,"_Ids_info_all.p"]), "rb" ) )
uni_gene = pickle.load(open( ''.join([name,"_uni_gene.p"]), "rb" ) )

PDBs_sel=[]
start_end_position = []
for ids in Ids_info_all:
    PDB_t =[]
    start_end_position_t =[]
    for id_t in ids:
        if len(id_t) == 3:
            for j in range(0,len(id_t[2])):
                    if id_t[2][j]=="-":
                       if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > threshold_length:
                           if id_t[0] in site_sat_flatten:
                               PDB_t.append(id_t[0])
                               start_end_position_t.append([int(id_t[2][0:j]),int(id_t[2][j+1:len(id_t[2])])])
                           else:
                               print("SITE_not_satisfied: ",id_t[0])
    PDBs_sel.append(PDB_t)
    start_end_position.append(start_end_position_t)

#%  after threshold satisfication tested the select gene and PDB details
fin_PDB_ids = [] # finalised_PDB_ids 
fin_PDB_ids_start_end_position = [] # finalised_PDB_ids_start_end_position
fin_uni_gene = [] # finalised_uni_gene 
for i in range(0,len(PDBs_sel)): 
    if len(PDBs_sel[i]) > 0:   
        fin_PDB_ids.append(PDBs_sel[i])
        fin_PDB_ids_start_end_position.append(start_end_position[i])
        fin_uni_gene.append(uni_gene[i])  