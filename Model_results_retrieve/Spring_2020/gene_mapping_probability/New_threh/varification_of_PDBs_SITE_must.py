# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

import os
import pickle

os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Man_thresh/21_amino_acids/21-prop')
#%%
MAN_train_OG=pickle.load(open('OG_train_PDBs.p', "rb"))
man_over_OG=pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
#%%

os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Before_thresh/21_amino_acids/21-prop')

bef_train_OG=pickle.load(open('OG_train_PDBs.p', "rb"))
bef_over_OG=pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
#%%
os.chdir('/')
os.chdir('F:/scrach_cacs_1083/optimaly_tilted_21_quarter/Test/ONGO')
MAn_ONGO_train=os.listdir()
#%%
os.chdir('/')
os.chdir('F:/scrach_cacs_1083/Before_thresh_changed/optimaly_tilted_21_quarter/Test/ONGO')
bef_ONGO_train=os.listdir()

#%%
for pdb in MAn_ONGO_train:
    if pdb not in bef_ONGO_train:
        print(pdb)
        
if '721P.npy' in  MAn_ONGO_train:
    print('gyhugd')
#%%
os.chdir('/')
os.chdir('F:/scrach_cacs_1083/optimaly_tilted_21_quarter/')
man_test=pickle.load(open('test_labels_dic.p', "rb"))
