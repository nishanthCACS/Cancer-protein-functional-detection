# -*- coding: utf-8 -*-
"""
Created on %(20-May-2020) at 6.28pm

@author: %A.Nishanth C00294860

 first create the data for 10-fold dataset
"""
import os
import pickle
from copy import deepcopy

unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
os.chdir('/')
os.chdir(unigene_load_dir)
ONGO_train_gene_id=pickle.load(open("ONGO_train_gene_id.p", "rb"))  
#pickle.dump(, open("ONGO_test_gene_id.p", "wb"))  
#pickle.dump(TSG_train_gene_id, open("TSG_train_gene_id.p", "wb"))  
#pickle.dump(TSG_test_gene_id, open("TSG_test_gene_id.p", "wb"))  
#pickle.dump(Fusion_train_gene_id, open("Fusion_train_gene_id.p", "wb"))  
#pickle.dump(Fusion_test_gene_id, open("Fusion_test_gene_id.p", "wb"))  
Tier=1
num_amino_acids=21
if Tier==1 and num_amino_acids==21:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/New_K_fold'
    old_main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/K_fold'

#%%

#os.chdir('/')
#os.chdir('F:/scrach_cacs_1083/optimaly_tilted_21_quarter/10-fold')

#%%
unigene_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_',str(Tier),'/SITE/before_thresh/',str(num_amino_acids),'_aminoacid'])
os.chdir('/')
os.chdir(unigene_load_dir)
def load_PDBs_and_gene_details(name,train_test=False):
    '''
    This function extract the PDB and gene details and return
    '''
    satisfied_PDBs= pickle.load(open(''.join([name,"_satisfied_PDBs.p"]), "rb")) 
    if train_test:
        test_new_gene_list = pickle.load(open(''.join([name,"_satisfied_Gene_ID.p"]), "rb")) 
        return test_new_gene_list,satisfied_PDBs
    else:
        satisfied_PDBs=list(set(sum(satisfied_PDBs,[])))
        new_pdb_list=[] 
        for pdbs in satisfied_PDBs:
            new_pdb_list.append(''.join([pdbs,'.npy']))
        return new_pdb_list

#%% first create the data for 10-fold dataset
ONGO_pdbs=load_PDBs_and_gene_details('ONGO')
TSG_pdbs=load_PDBs_and_gene_details('TSG')
Fusion_pdbs=load_PDBs_and_gene_details('Fusion')
#%%
labels=[]
'''TSG PDBs labeled as 0s '''
list_IDs=deepcopy(ONGO_pdbs)
for i in range(0,len(ONGO_pdbs)):
    labels.append(0)

'''TSG PDBs labeled as 1s '''
for i in range(0,len(TSG_pdbs)):
    labels.append(1)
    list_IDs.append(TSG_pdbs[i])

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
#%% go through the older PDB details and find out which PDBs has to be placed
os.chdir('/')
os.chdir(old_main_directory)
train_list_IDs_OLD = pickle.load(open("train_list_ids.p", "rb"))
train_labels_OLD = pickle.load(open("train_labels.p", "rb"))
PDB_needed=[]
for pdb in train_list_IDs:
    if pdb not in train_list_IDs_OLD:
        PDB_needed.append(pdb)