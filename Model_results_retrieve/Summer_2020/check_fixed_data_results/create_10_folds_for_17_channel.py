# -*- coding: utf-8 -*-
"""
Created on %24-May-2020 at 8.01pm

@author: %A.Nishanth C00294860

This will create the dataset of 10-folds of 17 channels(from 21 channel 10-fold results)
Just need to remove "5WLB" PDB from the folds
"""

import os
import pickle
sel_dir='F:/scrach_cacs_1083/10_fold_choosen'
#%%
os.chdir('/')
os.chdir(sel_dir)
fold= pickle.load(open('pdbs_valid_fold_0.p', "rb"))
#fold= pickle.load(open('pdbs_valid_fold_1.p', "rb"))
'''It will only print to fold 0'''
for group in fold:
    if ''.join(['5WLB','.npy']) in group:
        print('here')
        
#%%
        
for i in range(0,len(fold)):
    group=fold[i]
    if '5WLB.npy' in group:
        print('here: ',i)
        
#%% then take the 7th group and place the id from last group to make it 10
group_reassign=[]   
for i in range(0,len(fold)):
    group=fold[i]
    if i==len(fold)-1:
        pdb_insert=group.pop()
#%%
from copy import deepcopy
group_reassign=[]   
for i in range(0,len(fold)):
    group=fold[i]
    if i==7:
        pdb_insert=group.remove('5WLB.npy')
        print("popped")
        group.append(pdb_insert)
    group_reassign.append(deepcopy(group))
    
#%%
sel_dir='F:/scrach_cacs_1083/10_fold_choosen/17_Channels'

os.chdir('/')
os.chdir(sel_dir) 
pickle.dump(group_reassign, open("pdbs_valid_fold_0.p", "wb"))  
