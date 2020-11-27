# -*- coding: utf-8 -*-
"""
Created on %26-January-2018(15.52pm)


@author: %A.Nishanth C00294860
loading the experiment type details
"""
#import pickle
import os
import copy
import pickle
import shutil
#%%
"""
Since the cleaned data have some issues check agin and create clean data
"""

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
name = "ONGO"
os.chdir("/")
os.chdir(loading_dir)
ONGO_clean_unigene_details = pickle.load(open( ''.join([name,"_clean_UniGene.p"]), "rb" ) )
ONGO_thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
ONGO_thres_sat_pdb_list = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
name = "TSG"
TSG_clean_unigene_details = pickle.load(open( ''.join([name,"_clean_UniGene.p"]), "rb" ) )
TSG_thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
TSG_thres_sat_pdb_list = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
name = "Fusion"
Fusion_clean_unigene_details = pickle.load(open( ''.join([name,"_clean_UniGene.p"]), "rb" ) )
Fusion_thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
Fusion_thres_sat_pdb_list = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
#%% First check overlapping Unigene
def gene_chk(list_1,list_2):
    for u_g in list_1:
        if u_g in list_2:
            print(u_g)
gene_chk(ONGO_thres_sat_gene_list,TSG_thres_sat_gene_list)
gene_chk(ONGO_thres_sat_gene_list,Fusion_thres_sat_gene_list)
gene_chk(TSG_thres_sat_gene_list,Fusion_thres_sat_gene_list)
#%% then check the overlapping PDB_ids
def finding_problemtric_pdbs(PDB_given_1,PDB_given_2,gene_given_1,gene_given_2):
    """
    This function find the overlappin pdbs and their corresponding Gene details
    """
    pdb_list_1= list(set(sum(PDB_given_1, [])))
    pdb_list_2= list(set(sum(PDB_given_2, [])))
    fell_in_both_group = list(set(pdb_list_1) & set(pdb_list_2))
    problemetric_gene_1 = []
    for pdb in fell_in_both_group:
        for i in range(0,len(PDB_given_1)):
            if pdb in PDB_given_1[i]:
                problemetric_gene_1.append(gene_given_1[i])
    problemetric_gene_2 = []
    for pdb in fell_in_both_group:
        for i in range(0,len(PDB_given_2)):
            if pdb in PDB_given_2[i]:
                problemetric_gene_2.append(gene_given_2[i])
    
#    problemetric_gene_1 = list(set(problemetric_gene_1))
#    problemetric_gene_2 = list(set(problemetric_gene_2))
    return problemetric_gene_1, problemetric_gene_2, fell_in_both_group

problemetric_gene_ONGO_T, problemetric_gene_TSG_O, fell_in_both_group_O_T = finding_problemtric_pdbs(ONGO_thres_sat_pdb_list, TSG_thres_sat_pdb_list,ONGO_thres_sat_gene_list,TSG_thres_sat_gene_list)
problemetric_gene_ONGO_F, problemetric_gene_Fusion_O, fell_in_both_group_O_F = finding_problemtric_pdbs(ONGO_thres_sat_pdb_list, Fusion_thres_sat_pdb_list,ONGO_thres_sat_gene_list,Fusion_thres_sat_gene_list)
problemetric_gene_Fusion_T, problemetric_gene_TSG_F, fell_in_both_group_T_F = finding_problemtric_pdbs(Fusion_thres_sat_pdb_list, TSG_thres_sat_pdb_list,Fusion_thres_sat_gene_list,TSG_thres_sat_gene_list)
#%%
import numpy as np
"""

Then count the occurance of the problemtric Gene Occurance
"""
def count_prob_gene_occurance(problemetric_gene_1,problemetric_gene_2):
    unique_problemetric_gene_1 = list(set(problemetric_gene_1))
    unique_problemetric_gene_2 = list(set(problemetric_gene_2))
    count_prob_gene_1 = np.zeros((len(unique_problemetric_gene_1),1))
    count_prob_gene_2 = np.zeros((len(unique_problemetric_gene_2),1))
    for i in range(0,len(unique_problemetric_gene_1)):
        for g in problemetric_gene_1:
            if unique_problemetric_gene_1[i]==g:
                count_prob_gene_1[i] = count_prob_gene_1[i] + 1
    for i in range(0,len(unique_problemetric_gene_2)):
        for g in problemetric_gene_2:
            if unique_problemetric_gene_2[i]==g:
                count_prob_gene_2[i] = count_prob_gene_2[i] + 1            
    return count_prob_gene_1,count_prob_gene_2      

count_prob_gene_ONGO_T,count_prob_gene_TSG_O =  count_prob_gene_occurance(problemetric_gene_ONGO_T, problemetric_gene_TSG_O)
count_prob_gene_ONGO_F,count_prob_gene_Fusion_O =  count_prob_gene_occurance(problemetric_gene_ONGO_F, problemetric_gene_Fusion_O)
count_prob_gene_TSG_F,count_prob_gene_Fusion_T =  count_prob_gene_occurance(problemetric_gene_TSG_F, problemetric_gene_Fusion_T)
#%% Then check hpowmany percentage of the UniGenes overlapped from the selected PDBs
import copy
def index_problemetric_retrieve(problemetric_gene_list_1,problemetric_gene_list_2,thresh_sat_gene_list,thresh_sat_pdb_list):
    problemetric_gene_list = problemetric_gene_list_1 + problemetric_gene_list_2
    unique_problemetric_gene_list = list(set(problemetric_gene_list))
    index_problemetric = []
    for u_g in unique_problemetric_gene_list:
        for i in range(0,len(thresh_sat_gene_list)):
            if thresh_sat_gene_list[i] == u_g:
                index_problemetric.append(i)
    index_problemetric.sort(reverse=True)
    filtered_gene = copy.deepcopy(thresh_sat_gene_list)
    filtered_pdb = copy.deepcopy(thresh_sat_pdb_list)
    for i in index_problemetric:
        del filtered_gene[i]
        print(thresh_sat_gene_list[i],' is deleted corresponding index is ',i)
        print(filtered_pdb[i])
        del filtered_pdb[i]
        print(len(thresh_sat_pdb_list[i]),' pdbs are deleted')
    return filtered_gene,filtered_pdb

filtered_ONGO_gene,filtered_ONGO_pdb = index_problemetric_retrieve(problemetric_gene_ONGO_F,problemetric_gene_ONGO_T,ONGO_thres_sat_gene_list,ONGO_thres_sat_pdb_list)
print("")
print("--------------- cleaning TSG -----------------------")
print("")
filtered_TSG_gene,filtered_TSG_pdb = index_problemetric_retrieve(problemetric_gene_TSG_F,problemetric_gene_TSG_O,TSG_thres_sat_gene_list,TSG_thres_sat_pdb_list)
print("")
print("--------------- cleaning Fusion -----------------------")
print("")
filtered_Fusion_gene,filtered_Fusion_pdb = index_problemetric_retrieve(problemetric_gene_Fusion_T,problemetric_gene_Fusion_O,Fusion_thres_sat_gene_list,Fusion_thres_sat_pdb_list)
#%% then save the clean PDBs as pikles
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
os.chdir("/")
os.chdir(saving_dir)

pickle.dump(filtered_ONGO_gene, open("ONGO_clean_unigene.p", "wb" ))
pickle.dump(filtered_ONGO_pdb, open("ONGO_clean_PDBs.p", "wb" ))
pickle.dump(filtered_TSG_gene, open("TSG_clean_unigene.p", "wb" ))
pickle.dump(filtered_TSG_pdb, open("TSG_clean_PDBs.p", "wb" ))
pickle.dump(filtered_Fusion_gene, open("Fusion_clean_unigene.p", "wb" ))
pickle.dump(filtered_Fusion_pdb, open("Fusion_clean_PDBs.p", "wb" ))
#%% 
"""
Only considering the TSG Vs ONGO leave the Fusion intersection among the ONGO,TSG
Since, the Fusion class is made by the combination of ONGO and TSG, then Mutation may occur among that 
"""
def index_problemetric_retrieve(problemetric_gene_list,thresh_sat_gene_list,thresh_sat_pdb_list):
    unique_problemetric_gene_list = list(set(problemetric_gene_list))
    index_problemetric = []
    for u_g in unique_problemetric_gene_list:
        for i in range(0,len(thresh_sat_gene_list)):
            if thresh_sat_gene_list[i] == u_g:
                index_problemetric.append(i)
    index_problemetric.sort(reverse=True)
    filtered_gene = copy.deepcopy(thresh_sat_gene_list)
    filtered_pdb = copy.deepcopy(thresh_sat_pdb_list)
    for i in index_problemetric:
        del filtered_gene[i]
        print(thresh_sat_gene_list[i],' is deleted corresponding index is ',i)
        print(filtered_pdb[i])
        del filtered_pdb[i]
        print(len(thresh_sat_pdb_list[i]),' pdbs are deleted')
    return filtered_gene,filtered_pdb
print("")
print("--------------- cleaning ONGO -----------------------")
print("")
filtered_ONGO_gene,filtered_ONGO_pdb = index_problemetric_retrieve(problemetric_gene_ONGO_T,ONGO_thres_sat_gene_list,ONGO_thres_sat_pdb_list)
print("")
print("--------------- cleaning TSG -----------------------")
print("")
filtered_TSG_gene,filtered_TSG_pdb = index_problemetric_retrieve(problemetric_gene_TSG_O,TSG_thres_sat_gene_list,TSG_thres_sat_pdb_list)
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
os.chdir("/")
os.chdir(saving_dir)

pickle.dump(filtered_ONGO_gene, open("ONGO_clean_O_T_unigene.p", "wb" ))
pickle.dump(filtered_ONGO_pdb, open("ONGO_clean_O_T_PDBs.p", "wb" ))
pickle.dump(filtered_TSG_gene, open("TSG_clean_O_T_unigene.p", "wb" ))
pickle.dump(filtered_TSG_pdb, open("TSG_clean_O_T_PDBs.p", "wb" ))
#%% checking Fall_2018 and Spring 2019 cleaned genes are same for TSG and ONGO only considerd

import pickle
import os
spring_2019_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
os.chdir("/")
os.chdir(spring_2019_dir)
ONGO_clean_O_T_unigene = pickle.load(open( "ONGO_clean_O_T_unigene.p", "rb" ) )
ONGO_clean_O_T_PDBs = pickle.load(open("ONGO_clean_O_T_PDBs.p", "rb" ) )
TSG_clean_O_T_unigene = pickle.load(open("TSG_clean_O_T_unigene.p", "rb" ) )
TSG_clean_O_T_PDBs = pickle.load(open("TSG_clean_O_T_PDBs.p", "rb" ) )
#%%
#fall_2018_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
#os.chdir("/")
#os.chdir(fall_2018_dir)
##ONGO_clean_unigene = pickle.load(open( "ONGO_clean_UniGene.p", "rb" ) )
#ONGO_clean_PDBs = pickle.load(open( "ONGO_clean_unigene_PDB.p","rb" ) )
#TSG_clean_unigene = pickle.load(open( "TSG_clean_UniGene.p", "rb" ) )
#TSG_clean_PDBs = pickle.load(open( "TSG_clean_unigene_PDB.p","rb" ) )