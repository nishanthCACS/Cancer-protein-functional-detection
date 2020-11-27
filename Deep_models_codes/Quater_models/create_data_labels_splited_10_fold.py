#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:17:22 2019

@author: c00294860
"""
import os
from copy import deepcopy
import pickle
'''This python script is created to make balanced dataset'''
#%%
#main_directory =''.join(['scratch/before_thresh_change/optimaly_tilted_21_quarter_k_fold/'])
main_directory =''.join(['scratch/before_thresh_change/optimaly_tilted_17_quarter_k_fold/'])
ONGO_dir=''.join([main_directory,'Train/ONGO'])
TSG_dir= ''.join([main_directory,'Train/TSG'])
Fusion_dir=''.join([main_directory,'Train/Fusion'])

labels=[]
main_dir='scratch/neg_size_20_21_aminoacids/'

'''ONGO PDBs labeled as 0s '''
os.chdir('/')
os.chdir(ONGO_dir)

ONGO_pdbs=os.listdir()
list_IDs=deepcopy(ONGO_pdbs)
for i in range(0,len(ONGO_pdbs)):
    labels.append(0)

'''TSG PDBs labeled as 1s '''
os.chdir('/')
os.chdir(TSG_dir)

TSG_pdbs=os.listdir()
for i in range(0,len(TSG_pdbs)):
    labels.append(1)
    list_IDs.append(TSG_pdbs[i])

'''Fusion PDBs labeled as 1s '''
os.chdir('/')
os.chdir(Fusion_dir)

Fusion_pdbs=os.listdir()
for i in range(0,len(Fusion_pdbs)):
    labels.append(2)
    list_IDs.append(Fusion_pdbs[i])

os.chdir('/')
os.chdir(main_directory)
pickle.dump(list_IDs, open("train_list_ids.p", "wb"))  
pickle.dump(labels, open("train_labels.p", "wb"))  

train_list_IDs = pickle.load(open("train_list_ids.p", "rb"))
train_labels = pickle.load(open("train_labels.p", "rb"))
#%
train_labels_dic={}
for i in range(0,len(train_labels)):
    train_labels_dic.update({train_list_IDs[i]: train_labels[i]})

pickle.dump(train_labels_dic, open("train_labels_dic.p", "wb"))  


    
os.chdir('/')
os.chdir(main_directory)
train_list_IDs = pickle.load(open("train_list_ids.p", "rb"))
train_labels = pickle.load(open("train_labels.p", "rb"))
train_labels_dic = pickle.load(open("train_labels_dic.p", "rb"))  
#%     check howmany ONGO and TSG overlap with Fusion

ongo_chk=0
tsg_chk=0
overlapped_PDBs_ONGO=[]
overlapped_PDBs_TSG=[]
overlapped_PDBs=[]
for i in range(0,len(train_labels)):
    if train_labels[i]!=train_labels_dic[train_list_IDs[i]]:
        overlapped_PDBs.append(train_list_IDs[i])
        if train_labels[i]==0:
            overlapped_PDBs_ONGO.append(train_list_IDs[i])
            ongo_chk=ongo_chk+1
        elif train_labels[i]==1:
            overlapped_PDBs_TSG.append(train_list_IDs[i])
            tsg_chk=tsg_chk+1 
print('ONGO overlapped Fusion: ',ongo_chk)
print('TSG overlapped Fusion: ',tsg_chk)
# create seperate dictinary that only have no overlapping PDBs
clean_train_labels_dic={}
for i in range(0,len(train_labels)):
    if train_list_IDs[i] not in overlapped_PDBs:
        clean_train_labels_dic.update({train_list_IDs[i]: train_labels[i]})
#%
''' oneway is only using clean no overlapping with Fusion at all'''
OG_train_PDBs=[]
TSG_train_PDBs=[]
Fusion_train_PDBs=[]
for i in range(0,len(train_labels)):
    if (train_list_IDs[i] not in overlapped_PDBs) and train_labels[i]==0:
        OG_train_PDBs.append(train_list_IDs[i])
    elif  (train_list_IDs[i] not in overlapped_PDBs) and train_labels[i]==1:
        TSG_train_PDBs.append(train_list_IDs[i])
    elif  (train_list_IDs[i] not in overlapped_PDBs) and train_labels[i]==2:
        Fusion_train_PDBs.append(train_list_IDs[i])
#%% First split the PDB ids into groups  
'''Here the overlap only contains the PDBs with Fusion'''
pickle.dump(OG_train_PDBs, open("OG_train_PDBs.p", "wb"))  
pickle.dump(TSG_train_PDBs, open("TSG_train_PDBs.p", "wb"))      
pickle.dump(Fusion_train_PDBs, open("Fusion_train_PDBs.p", "wb"))  
pickle.dump(overlapped_PDBs_ONGO, open("overlapped_PDBs_ONGO.p", "wb"))      
pickle.dump(overlapped_PDBs_TSG, open("overlapped_PDBs_TSG.p", "wb"))      
