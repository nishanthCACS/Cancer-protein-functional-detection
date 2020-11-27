# -*- coding: utf-8 -*-
"""
Created on %(20-May-2020) at 6.28pm

@author: %A.Nishanth C00294860

 first create the data for 10-fold dataset
"""
import os
import pickle
from copy import deepcopy

Tier=1
num_amino_acids=21
if Tier==1 and num_amino_acids==21:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/New_K_fold'
    old_main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/K_fold'

#%%
def problemtric_chk_rem(given_list,gene_list_sat,overlapped_genes):
    '''
    To make sure again SITE satisfied data only taken
    '''
    new_formed=[]
    for gene in given_list:
        if gene in gene_list_sat:
            if gene not in overlapped_genes:
                new_formed.append(gene)
    return new_formed

def helper_for_finalise_PDBS(given_list,gene_list_all,satisfied_PDBs):
    sel_PDBS=[]
    for i in range(0,len(gene_list_all)):
        if gene_list_all[i] in given_list:
            sel_PDBS.append(deepcopy(satisfied_PDBs[i]))
    sel_PDBS= list(set(sum(sel_PDBS,[])))
    fin_sel_PDBs=[]
    for pdbs in sel_PDBS:
        fin_sel_PDBs.append(''.join([pdbs,'.npy']))
    return fin_sel_PDBs

def load_PDBs_and_gene_details(name,train_test=False):
    '''
    This function extract the PDB and gene details and return
    To make sure again SITE satisfied data only taken

    '''        
    All_load_dir ='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene/'
    os.chdir('/')
    os.chdir(All_load_dir)
    SITE_satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
    SITE_satisfied_gene_list_all= pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb"))
    
    SITE_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_',str(Tier),'/SITE/before_thresh/',str(num_amino_acids),'_aminoacid'])
    os.chdir('/')
    os.chdir(SITE_load_dir)
    satisfied_PDBs= pickle.load(open(''.join([name,"_satisfied_PDBs.p"]), "rb")) 
    gene_list_all= pickle.load(open(''.join([name,"_satisfied_Gene_ID.p"]), "rb"))
    if train_test:
        '''
        To load the train test details
        The genes here are only mapped from 
        '''
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        if name=="Fusion":
           train_gene_id =pickle.load(open(''.join([name,"_train_gene_id.p"]), "rb"))  
           test_gene_id =pickle.load(open(''.join([name,"_test_gene_id.p"]), "rb"))  
        else:
           train_gene_id_t =pickle.load(open(''.join([name,"_train_gene_id.p"]), "rb"))  
           test_gene_id_t =pickle.load(open(''.join([name,"_test_gene_id.p"]), "rb"))  
           overlapped_genes_t= pickle.load(open(''.join(['problemteric_',name,'_genes.p']), "rb")) 
           overlapped_genes=[]
           for o_g in overlapped_genes_t:
               overlapped_genes.append(o_g[2])
           
           train_gene_id = problemtric_chk_rem(train_gene_id_t,gene_list_all,overlapped_genes)
           test_gene_id = problemtric_chk_rem(test_gene_id_t,gene_list_all,overlapped_genes)

           overlapped_gene_PDBs=helper_for_finalise_PDBS(overlapped_genes,SITE_satisfied_gene_list_all,SITE_satisfied_PDBs)

        train_sat_PDBs=helper_for_finalise_PDBS(train_gene_id,gene_list_all,satisfied_PDBs)
        test_sat_PDBs=helper_for_finalise_PDBS(test_gene_id,gene_list_all,satisfied_PDBs)

        pickle.dump(train_gene_id,open(''.join([name,"_train_genes_21.p"]), "wb")) 
        pickle.dump(test_gene_id,open(''.join([name,"_test_genes_21.p"]), "wb")) 

        if name=="Fusion":
            return train_sat_PDBs,test_sat_PDBs
        else:   
            pickle.dump(overlapped_genes,open(''.join([name,"_overlapped_genes_21.p"]), "wb")) 
            return train_sat_PDBs,test_sat_PDBs,overlapped_gene_PDBs#,train_gene_id,test_gene_id
    else:
        satisfied_PDBs=list(set(sum(satisfied_PDBs,[])))
        new_pdb_list=[] 
        for pdbs in satisfied_PDBs:
            new_pdb_list.append(''.join([pdbs,'.npy']))
        return new_pdb_list

#%% first create the data for 10-fold dataset
#load_PDBs_and_gene_details('ONGO')

ONGO_train_sat_PDBs,ONGO_test_sat_PDBs,ONGO_overlapped_gene_PDBs=load_PDBs_and_gene_details('ONGO',train_test=True)
TSG_train_sat_PDBs,TSG_test_sat_PDBs,TSG_overlapped_gene_PDBs=load_PDBs_and_gene_details('TSG',train_test=True)
Fusion_train_sat_PDBs,Fusion_test_sat_PDBs=load_PDBs_and_gene_details('Fusion',train_test=True)

#%%
def pdbs_to_label_IDS(ONGO_pdbs,TSG_pdbs,Fusion_pdbs):
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
    
    return labels,list_IDs


train_labels,train_list_IDs=pdbs_to_label_IDS(ONGO_train_sat_PDBs,TSG_train_sat_PDBs,Fusion_train_sat_PDBs)
test_labels,test_list_IDs=pdbs_to_label_IDS(ONGO_test_sat_PDBs,TSG_test_sat_PDBs,Fusion_test_sat_PDBs)

main_dir='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/V_91_train_test_pickles'
#main_dir='E:/BIBM_project_data/V_91_21_channels_bef_thresh'
os.chdir('/')
os.chdir(main_dir)
pickle.dump(train_list_IDs, open("train_list_ids.p", "wb"))  
pickle.dump(train_labels, open("train_labels.p", "wb"))  

#%
train_labels_dic={}
for i in range(0,len(train_labels)):
    train_labels_dic.update({train_list_IDs[i]: train_labels[i]})
test_labels_dic={}
for i in range(0,len(test_labels)):
    test_labels_dic.update({test_list_IDs[i]: test_labels[i]})
pickle.dump(train_labels_dic, open("train_labels_dic.p", "wb")) 

 
pickle.dump(test_list_IDs, open("test_list_ids.p", "wb"))  
pickle.dump(test_labels_dic, open("test_labels_dic.p", "wb"))  

train_list_IDs = pickle.load(open("train_list_ids.p", "rb"))
train_labels = pickle.load(open("train_labels.p", "rb"))
train_labels_dic = pickle.load(open("train_labels_dic.p", "rb"))  

#%%     check howmany ONGO and TSG overlap with Fusion
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
#%%
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

pickle.dump(ONGO_overlapped_gene_PDBs, open("ONGO_overlapped_gene_PDBs.p", "wb"))     
pickle.dump(TSG_overlapped_gene_PDBs, open("TSG_overlapped_gene_PDBs.p", "wb"))      
 

#% first check where these files exist
#old_main_dir='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/Train/ONGO/'
#os.chdir('/')
#os.chdir(old_main_dir)
#old_files=os.listdir()
#count=0
#pdb_there=[]
#for pdb in old_files:
#    if pdb in test_PDB_needed:
#        pdb_there.append(pdb)
##%
##%%  
#
#import shutil
#main_dir='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/V_91_train_test_pickles/test_needed'
#
#dir_src = (old_main_dir)
#dir_dst = (main_dir)
#
#for pdb in test_PDB_needed:
#    shutil.move(dir_src + pdb, dir_dst)
#    print(pdb)       
