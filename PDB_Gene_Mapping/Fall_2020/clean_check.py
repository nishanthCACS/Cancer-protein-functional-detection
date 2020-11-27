# -*- coding: utf-8 -*-
"""
Created on %(20-May-2020) at 6.28pm
Rechecked on 26-Sep-2020(08.47Am)

@author: %A.Nishanth C00294860
"""
import os
import pickle
from copy import deepcopy


Tier=1
num_amino_acids=21
if Tier==1 and num_amino_acids==21:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/New_K_fold'
    old_main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/K_fold'

#
unigene_load_dir =     'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene'
#unigene_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_',str(Tier),'/SITE/before_thresh/',str(num_amino_acids),'_aminoacid'])
os.chdir('/')
os.chdir(unigene_load_dir)
#name="ONGO"
#satisfied_PDBs= pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb")) 
#satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
def load_PDBs_and_gene_details(name,train_test=True):
    '''
    This function extract the PDB and gene details and return
    '''
    satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
    if train_test:
        test_new_gene_list = pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb")) 
        return test_new_gene_list,satisfied_PDBs
    else:
        satisfied_PDBs=list(set(sum(satisfied_PDBs,[])))
        new_pdb_list=[] 
        for pdbs in satisfied_PDBs:
            new_pdb_list.append(''.join([pdbs,'.npy']))
        return new_pdb_list

#%% first create the data for 10-fold dataset
ONGO_genes,ONGO_pdbs=load_PDBs_and_gene_details('ONGO')
TSG_genes,TSG_pdbs=load_PDBs_and_gene_details('TSG')
Fusion_genes,Fusion_pdbs=load_PDBs_and_gene_details('Fusion')

#%%
def load_PDBs_and_gene_details_dic(name):
    '''
    This function extract the PDB and gene details and return
    '''
    satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
    gene_PDB_dic={}
    test_new_gene_list = pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb")) 

    for i in range(0,len(satisfied_PDBs)):
        if len(satisfied_PDBs[i])>0:
            gene_PDB_dic[test_new_gene_list[i]]=satisfied_PDBs[i]
    return gene_PDB_dic
ONGO_gene_pdbs=load_PDBs_and_gene_details_dic('ONGO')
TSG_gene_pdbs=load_PDBs_and_gene_details_dic('TSG')
Fusion_gene_pdbs=load_PDBs_and_gene_details_dic('Fusion')   

#%%
ONGO_pdbs_sum=list(set(sum(ONGO_pdbs,[])))

TSG_pdbs_sum=list(set(sum(TSG_pdbs,[])))
Fusion_pdbs_sum=list(set(sum(Fusion_pdbs,[])))
#%%
'''
Cehcking the example case for paper
'''
def ovrelap_chk(chk_group,cls_1_pdbs_sum,cls_2_pdbs_sum):
    count_cls_1_only=0
    count_cls_2_only=0
    count_cls_1_cls_2=0
    for pdb in chk_group:
        cls_1_chk=False
        cls_2_chk=False
        if pdb in cls_1_pdbs_sum:
            cls_1_chk=True
        if pdb in cls_2_pdbs_sum:
            cls_2_chk=True
        if cls_1_chk & cls_2_chk:
           count_cls_1_cls_2=count_cls_1_cls_2+1
        elif cls_1_chk:
            count_cls_1_only=count_cls_1_only+1
        elif cls_2_chk:
            count_cls_2_only=count_cls_2_only+1
    return count_cls_1_only,count_cls_2_only,count_cls_1_cls_2
print(len(ONGO_gene_pdbs['3020']))
print(ovrelap_chk(ONGO_gene_pdbs['3020'],TSG_pdbs_sum,Fusion_pdbs_sum))
#%%
print(len(ONGO_gene_pdbs['8350']))
print(ovrelap_chk(ONGO_gene_pdbs['8350'],TSG_pdbs_sum,Fusion_pdbs_sum))
#%%
print(len(ONGO_gene_pdbs['5290']))
print(ovrelap_chk(ONGO_gene_pdbs['5290'],TSG_pdbs_sum,Fusion_pdbs_sum))
#%%
print(len(ONGO_gene_pdbs['7514']))
print(ovrelap_chk(ONGO_gene_pdbs['7514'],TSG_pdbs_sum,Fusion_pdbs_sum))
#%%
print(len(TSG_gene_pdbs['5295']))
print(ovrelap_chk(TSG_gene_pdbs['5295'],ONGO_pdbs_sum,Fusion_pdbs_sum))
#%%
print(len(Fusion_gene_pdbs['3105']))
print(ovrelap_chk(Fusion_gene_pdbs['3105'],ONGO_pdbs_sum,TSG_pdbs_sum))
