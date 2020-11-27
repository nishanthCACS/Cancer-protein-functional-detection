# -*- coding: utf-8 -*-
"""
Created on %(20-May-2020) at 6.28pm

@author: %A.Nishanth C00294860

 first create the fully cleaned data for 10-fold dataset
"""
import os
import pickle
from copy import deepcopy


Tier=1
num_amino_acids=21
SITE=True
if Tier==1 and num_amino_acids==21:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/New_K_fold'
    old_main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/K_fold'

  
#%%

#name="ONGO"
#satisfied_PDBs= pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb")) 
#satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
def load_PDBs_and_gene_details(name,train_test=False,SITE=True):
    '''
    This function extract the PDB and gene details and return
    '''
    SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
    os.chdir('/')
    os.chdir(SITE_dir)
    # since both 21 amino acids and 20 amino acids not-satisfied in have same PDBs
    not_satisfied_model=pickle.load(open('not_satisfied_model.p', "rb"))  #the PDB data unable to convert to supported quarter model formation 

    if SITE:
        unigene_load_dir =     'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        test_new_gene_list= pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb")) 
        satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 


    if train_test:
        return test_new_gene_list,satisfied_PDBs
    else:
        satisfied_PDBs=list(set(sum(satisfied_PDBs,[])))
        new_pdb_list=[] 
        for pdbs in satisfied_PDBs:
            if pdbs not in not_satisfied_model:
                new_pdb_list.append(''.join([pdbs,'.npy']))
#        return new_pdb_list
        return satisfied_PDBs
#%% first create the data for 10-fold dataset
ONGO_pdbs=load_PDBs_and_gene_details('ONGO')
TSG_pdbs=load_PDBs_and_gene_details('TSG')
Fusion_pdbs=load_PDBs_and_gene_details('Fusion')
#%%
train_labels=[]
'''TSG PDBs labeled as 0s '''
train_list_IDs=deepcopy(ONGO_pdbs)
for i in range(0,len(ONGO_pdbs)):
    train_labels.append(0)

'''TSG PDBs labeled as 1s '''
for i in range(0,len(TSG_pdbs)):
    train_labels.append(1)
    train_list_IDs.append(TSG_pdbs[i])

for i in range(0,len(Fusion_pdbs)):
    train_labels.append(2)
    train_list_IDs.append(Fusion_pdbs[i])
#%
train_labels_dic={}
for i in range(0,len(train_labels)):
    train_labels_dic.update({train_list_IDs[i]: train_labels[i]})


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
        
#%% Finding the genes whcih has more overlap in ONGO with TSG

unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene'
os.chdir('/')
os.chdir(unigene_load_dir)
ONGO_gene_list= pickle.load(open("ONGO_Gene_ID.p", "rb")) 
ONGO_satisfied_PDBs= pickle.load(open("ONGO_SITE_satisfied.p", "rb")) 
TSG_satisfied_PDBs= pickle.load(open("TSG_SITE_satisfied.p", "rb")) 
TSG_gene_list= pickle.load(open("TSG_Gene_ID.p", "rb")) 

overlapped_PDBs=overlapped_PDBs_ONGO+overlapped_PDBs_TSG
problemteric_ONGO=[]
for i in range(0,len(ONGO_satisfied_PDBs)):
    ongo_chk=0
    pdb_group=ONGO_satisfied_PDBs[i]
    for pdb in pdb_group:
        if pdb in overlapped_PDBs:
            ongo_chk=ongo_chk+1
    if ongo_chk>len(pdb_group)/2:
        problemteric_ONGO.append([len(pdb_group),ongo_chk,ONGO_gene_list[i]])
problemteric_TSG=[]
for i in range(0,len(TSG_satisfied_PDBs)):
    tsg_chk=0
    pdb_group=TSG_satisfied_PDBs[i]
    for pdb in pdb_group:
        if pdb in overlapped_PDBs:
            tsg_chk=tsg_chk+1
    if tsg_chk>len(pdb_group)/2:
        problemteric_TSG.append([len(pdb_group),tsg_chk,TSG_gene_list[i]])
pickle.dump(problemteric_TSG, open("problemteric_TSG_genes.p", "wb"))  
pickle.dump(problemteric_ONGO, open("problemteric_ONGO_genes.p", "wb"))  
