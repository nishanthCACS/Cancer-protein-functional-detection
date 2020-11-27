# -*- coding: utf-8 -*-
"""
Created on %(01-Feb-2020) 11.07Pm

@author: %A.Nishanth C00294860
"""
import os
import pickle
gene_sep_train_test ='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated'
#varify the dataset in 10-fola and the other one is same
os.chdir('/')
os.chdir(gene_sep_train_test)

#for organise the results or performance
OG_train_PDBs=  pickle.load(open("OG_train_PDBs.p", "rb"))
TSG_train_PDBs=  pickle.load(open("TSG_train_PDBs.p", "rb"))    
Fusion_train_PDBs=  pickle.load(open("Fusion_train_PDBs.p", "rb"))
test_list_ids = pickle.load(open("test_list_ids.p", "rb"))
overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
test_labels_dic= pickle.load(open("test_labels_dic.p", "rb"))

OG_test=[]
for ids in test_list_ids:
    if test_labels_dic[ids]==0:
        OG_test.append(ids)
OG_all=OG_train_PDBs+OG_test
OG_all_clean=[]
for ids in OG_all:
    if ids not in overlapped_PDBs_ONGO:
        if ids[0:4] not in OG_all_clean:
            OG_all_clean.append(ids[0:4])
TSG_test=[]
for ids in test_list_ids:
    if test_labels_dic[ids]==1:
        TSG_test.append(ids)
TSG_all=TSG_train_PDBs+TSG_test
TSG_all_clean=[]
for ids in TSG_all:
    if ids not in overlapped_PDBs_TSG:
        if ids[0:4] not in TSG_all_clean:
            TSG_all_clean.append(ids[0:4])
            
Fusion_test=[]
for ids in test_list_ids:
    if test_labels_dic[ids]==2:
        Fusion_test.append(ids)
Fusion_all=Fusion_train_PDBs+Fusion_test

overlapped_PDBs_Fusion= overlapped_PDBs_ONGO+overlapped_PDBs_TSG
Fusion_all_clean=[]
for ids in Fusion_all:
    if ids not in overlapped_PDBs_Fusion:
        if ids[0:4] not in Fusion_all_clean:
            Fusion_all_clean.append(ids[0:4])            
#taking 10folds set
ten_fold = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results"
#%%
os.chdir('/')
os.chdir(ten_fold)
#for organise the results or performance
OG_train_PDBs=  pickle.load(open("OG_train_PDBs.p", "rb"))
TSG_train_PDBs=  pickle.load(open("TSG_train_PDBs.p", "rb"))    
Fusion_train_PDBs=  pickle.load(open("Fusion_train_PDBs.p", "rb"))

if len(OG_train_PDBs)==len(OG_all_clean):
    print("ONGO use same data in 10-fold and gene seperated")
else:
    raise("data fault ONGO")
if len(TSG_train_PDBs)==len(TSG_all_clean):
    print("TSG use same data in 10-fold and gene seperated")
else:
    raise("data fault TSG")
if len(Fusion_train_PDBs)==len(Fusion_all_clean):
    print("Fusion use same data in 10-fold and gene seperated")
else:
    raise("data fault Fusion")