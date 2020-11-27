# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import os
import pickle
main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results"

os.chdir('/')
os.chdir(main_dir)
#for organise the results or performance
overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
train_list_ids= pickle.load(open("train_list_ids.p", "rb"))
#test_labels_dic= pickle.load(open("train_labels_dic.p", "rb"))
#test_labels_dic['2HYY.npy']     
#test_labels_dic['121P.npy']        
OG_train_PDBs= pickle.load(open("OG_train_PDBs.p", "rb"))

#%%
if '2HYY.npy' in OG_train_PDBs:
    print("OK")