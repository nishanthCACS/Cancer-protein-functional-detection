# -*- coding: utf-8 -*-
"""
Created on %08-Nov-2020(6.26P.m)

@author: %A.Nishanth C00294860
"""

import os
import pickle
from copy import deepcopy
# first load the fixed dataset of BIR model
#sel_dir='E:\scrach_cacs_1083\bef_thresh_21\10_fold_choosen_fixed_data_set\21_channels'
sel_dir='C:/Users/nishy/Desktop/10_fold_choosen_fixed_data_set'
def chk_in_the_given(fold,given_pdb_list):

    all_pdbs=[]
    for pdb_list in fold:
        temp=[]
        for pdb in pdb_list:
            if pdb in given_pdb_list:
                temp.append(pdb)
            else:
                print("skipped " ,pdb)
        all_pdbs.append(deepcopy(temp))
    return all_pdbs

os.chdir('/')
os.chdir(sel_dir)
ONGO_fold= pickle.load(open('10_splited_clean_ONGO.p', "rb"))
TSG_fold= pickle.load(open('10_splited_clean_TSG.p', "rb"))
Fusion_fold= pickle.load(open('10_splited_clean_Fusion.p', "rb"))
#%%
os.chdir('/')
os.chdir('E:/20_prop_bef_round/floor')
ONGO_given= pickle.load(open('OG_train_PDBs.p', "rb"))
TSG_given= pickle.load(open('TSG_train_PDBs.p', "rb"))
Fusion_given= pickle.load(open('Fusion_train_PDBs.p', "rb"))
#%%
ONGO_new=chk_in_the_given(ONGO_fold,ONGO_given)
TSG_new=chk_in_the_given(TSG_fold,TSG_given)
Fusion_new=chk_in_the_given(Fusion_fold,Fusion_given)

pickle.dump(ONGO_new, open('10_splited_clean_ONGO.p', "wb"))  
pickle.dump(TSG_new, open('10_splited_clean_TSG.p', "wb"))  
pickle.dump(Fusion_new, open('10_splited_clean_Fusion.p', "wb"))  
