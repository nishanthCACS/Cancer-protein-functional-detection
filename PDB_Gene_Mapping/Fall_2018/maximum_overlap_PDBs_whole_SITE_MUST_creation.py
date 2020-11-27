# -*- coding: utf-8 -*-
"""
Created on %18-November-2018(09.17pm)

@author: %A.Nishanth C00294860
"""
import pickle
import os
#import copy

traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
#name = "ONGO"
def retrieve_ids(name,traing_testing_pdb_dir):
    ongo_uni_gene_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
    os.chdir("/")
    os.chdir(ongo_uni_gene_pdb_dir)
    unigene = pickle.load(open(''.join([name,"_thres_sat_gene_list.p"]), "rb" ) )
    unigene_PDB = pickle.load(open(''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ) )
    return unigene,unigene_PDB

OG_unigene,OG_unigene_PDB =  retrieve_ids("ONGO",traing_testing_pdb_dir)
TSG_unigene,TSG_unigene_PDB =  retrieve_ids("TSG",traing_testing_pdb_dir)
Fusion_unigene,Fusion_unigene_PDB =  retrieve_ids("Fusion",traing_testing_pdb_dir)
#%% check the uniqueness of the UniGene name
"""checking the Uniqueness of the UniGene"""
ONGO_unique_unigene = list(set(OG_unigene))
TSG_unique_unigene = list(set(TSG_unigene))
Fusion_unique_unigene = list(set(Fusion_unigene))
""" since the Fusion UniGene has one alias, I place the UniGene id(Hs.713441) for training only"""

#%% then load the cleaned PDBs and remove the overlapping PDBs and UniGene
def load_clean_gene_pdbs(name,unigene,unigene_PDB):
    """
    load the pikles data of clened data information
    that means it remove the problemetric UniGene
    """
    if name == "ONGO" or name == "TSG":        
        uni_gene_problemetric = pickle.load(open(''.join([name,"_uni_gene_problemetric.p"]), "rb" ) )
    
    clean_UniGene =[]
    clean_only_unigene_PDBs = []
    for i in range(0,len(unigene)):
        if unigene[i]== uni_gene_problemetric[0]:
            print(name, unigene[i], " skipped")
        else:
            clean_UniGene.append(unigene[i])
            clean_only_unigene_PDBs.append(unigene_PDB[i])
    return clean_UniGene, clean_only_unigene_PDBs

uni_gene_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
os.chdir("/")
os.chdir(uni_gene_pdb_dir)
OG_clean_UniGene, OG_clean_only_unigene_PDBs = load_clean_gene_pdbs("ONGO",OG_unigene,OG_unigene_PDB)
TSG_clean_UniGene, TSG_clean_only_unigene_PDBs = load_clean_gene_pdbs("TSG",TSG_unigene,TSG_unigene_PDB)
Fusion_clean_UniGene = Fusion_unigene
Fusion_clean_only_unigene_PDBs = Fusion_unigene_PDB

#%%
"""
 From the results above only one id per classes in ONGO Vs TSG causes this problems 
 thus just remove those UniGenes and train the rest keep them seperately 
 for testing seperately
 
Data creation starts here!!!!!!!!!!!!!!!!!!

first load the PDB_ids used in new_ran_dataset for training and testing and remove the overlapping(unwanted) PDB_ids     
"""
def get_PDB_ids(name,unigene,unigene_PDB,created_train_test_list_dir):
    """
    Here
    
    unigene: cleaned unigene ids
    unigene_PDB: cleaned unigene to PDB_ids
    """
    os.chdir("/")
    os.chdir(created_train_test_list_dir)
    whole_training_unigene_temp = pickle.load(open(''.join([name,"_train_uni_gene.p"]), "rb" ) ) 
    # testing data pickle
    whole_testing_unigene_temp = pickle.load(open(''.join([name,"_test_uni_gene.p"]), "rb" ) ) 

#    whole_training_unigene_temp  = pickle.load(open(''.join([name,"_whole_train_unigene.p"]), "rb" ) ) 
#    # testing data pickle 
#    whole_testing_unigene_temp  = pickle.load(open(''.join([name,"_whole_test_unigene.p"]), "rb" ) ) 
    whole_training_unigene_temp = list(set(whole_training_unigene_temp))
    whole_training_pdb_ids = []
    whole_training_unigene = []
    for u in whole_training_unigene_temp:
        for i in range(0,len(unigene)):
                if unigene[i] == u:
#                    print("training_unigene: ",u)
                    whole_training_pdb_ids.append(unigene_PDB[i])
                    whole_training_unigene.append(u)
    whole_testing_pdb_ids= []
    whole_testing_unigene = []
    for u in whole_testing_unigene_temp:
        if u not in whole_training_unigene:
            for i in range(0,len(unigene)):
                if unigene[i] == u:
#                    print("test_unigene: ",u)
                    whole_testing_pdb_ids.append(unigene_PDB[i])    
                    whole_testing_unigene.append(u)
    return whole_training_pdb_ids, whole_testing_pdb_ids,whole_training_unigene,whole_testing_unigene
#created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
created_train_test_list_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/new_ran_whole_2"
#created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/new_random_whole"   
#created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/og_10" 
#OG_whole_training_pdb_ids, OG_whole_testing_pdb_ids,OG_whole_training_unigene, OG_whole_testing_unigene = get_PDB_ids("ONGO",OG_clean_UniGene,OG_clean_only_unigene_PDBs,created_train_test_list_dir)
#%%
#created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/new_random_whole"    
TSG_whole_training_pdb_ids, TSG_whole_testing_pdb_ids,TSG_whole_training_unigene, TSG_whole_testing_unigene = get_PDB_ids("TSG",TSG_clean_UniGene,TSG_clean_only_unigene_PDBs,created_train_test_list_dir)
#Fusion_whole_training_pdb_ids, Fusion_whole_testing_pdb_ids, Fusion_whole_training_unigene, Fusion_whole_testing_unigene = get_PDB_ids("Fusion",Fusion_clean_UniGene,Fusion_clean_only_unigene_PDBs,created_train_test_list_dir)
#%% then check the already created datasets to remove the unwanted PDBs
import shutil

def create_data_set(name,whole_training_pdb_ids,whole_testing_pdb_ids,whole_training_unigene,whole_testing_unigene):
    """
    Inorder to copy the files to copy the results
    
    """
    saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
    saving_dir_train =  ''.join([saving_pik, "/train/",name])
    saving_dir_test =  ''.join([saving_pik, "/test/",name])
    # check whether the apth exists or not and create the directory if not there
    if not os.path.isdir(saving_dir_train):
        os.makedirs(saving_dir_train)      
        print(saving_dir_train, " path created")
    if not os.path.isdir(saving_dir_test):
        os.makedirs(saving_dir_test)      
        print(saving_dir_test, " path created")
            
    train_pdbs = sum(whole_training_pdb_ids, [])
    test_pdbs = sum(whole_testing_pdb_ids, [])
    
    # since these surface results is already calculated so copy from that
    source_results_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/",name])
    os.chdir("/")
    os.chdir(source_results_dir)
    missed_pdbs_train =[]
    for pdb_name in train_pdbs:
        fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
        if os.path.isfile(fname): 
            file_xy = ''.join([pdb_name,'_xy.txt'])
            file_yz = ''.join([pdb_name,'_yz.txt'])
            file_xz = ''.join([pdb_name,'_xz.txt'])
            shutil.copy(file_xy, saving_dir_train)          
            shutil.copy(file_yz, saving_dir_train)    
            shutil.copy(file_xz, saving_dir_train)
        else:
            missed_pdbs_train.append(pdb_name)
    missed_pdbs_test = []
    for pdb_name in test_pdbs:
        fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
        if os.path.isfile(fname): 
            file_xy = ''.join([pdb_name,'_xy.txt'])
            file_yz = ''.join([pdb_name,'_yz.txt'])
            file_xz = ''.join([pdb_name,'_xz.txt'])
            shutil.copy(file_xy, saving_dir_test)          
            shutil.copy(file_yz, saving_dir_test)    
            shutil.copy(file_xz, saving_dir_test)
        else:
            missed_pdbs_test.append(pdb_name)
    #%save the created randomised training unigene pdb ids as pikle files
    os.chdir("/")
    os.chdir(saving_pik)
    pickle.dump(whole_training_unigene, open( ''.join([name,"_train_uni_gene.p"]), "wb" ))
    pickle.dump(whole_training_pdb_ids, open( ''.join([name,"_train_pdb_ids.p"]), "wb" ) )  
    pickle.dump(whole_testing_unigene, open( ''.join([name,"_test_uni_gene.p"]), "wb" ))
    pickle.dump(whole_testing_pdb_ids, open( ''.join([name,"_test_pdb_ids.p"]), "wb" ) )  
    print(name, " created :)")
    return missed_pdbs_train, missed_pdbs_test

#%% creating dataset with unoverlapping PDBs
#create_data_set("ONGO",OG_whole_training_pdb_ids, OG_whole_testing_pdb_ids,OG_whole_training_unigene, OG_whole_testing_unigene)
TSG_missed_pdbs_train, TSG_missed_pdbs_test  = create_data_set("TSG",TSG_whole_training_pdb_ids, TSG_whole_testing_pdb_ids,TSG_whole_training_unigene, TSG_whole_testing_unigene)
#create_data_set("Fusion",Fusion_whole_training_pdb_ids, Fusion_whole_testing_pdb_ids, Fusion_whole_training_unigene, Fusion_whole_testing_unigene)
#%% 
   """swapping with TSG"""

TSG_whole_training_unigene.append(TSG_whole_testing_unigene[12])   
TSG_whole_training_unigene.append(TSG_whole_testing_unigene[6]) 
TSG_whole_training_unigene.append(TSG_whole_testing_unigene[5]) 
TSG_whole_training_pdb_ids.append(TSG_whole_testing_pdb_ids[12])
TSG_whole_training_pdb_ids.append(TSG_whole_testing_pdb_ids[6])
TSG_whole_training_pdb_ids.append(TSG_whole_testing_pdb_ids[5])

#%%
TSG_whole_testing_unigene.append(TSG_whole_training_unigene[48])   
TSG_whole_testing_unigene.append(TSG_whole_training_unigene[47]) 
TSG_whole_testing_unigene.append(TSG_whole_training_unigene[6]) 
TSG_whole_testing_pdb_ids.append(TSG_whole_training_pdb_ids[48])
TSG_whole_testing_pdb_ids.append(TSG_whole_training_pdb_ids[47])
TSG_whole_testing_pdb_ids.append(TSG_whole_training_pdb_ids[6])
#%%    
"""then create the Dataset for any class seperately
then swap some the UniGenes for training set for ONGO"""

OG_whole_training_unigene.append(OG_whole_testing_unigene[13])   
#OG_whole_training_unigene.append(OG_whole_testing_unigene[12])  
OG_whole_training_unigene.append(OG_whole_testing_unigene[4]) 
OG_whole_training_unigene.append(OG_whole_testing_unigene[0]) 
OG_whole_training_pdb_ids.append(OG_whole_testing_pdb_ids[13])   
#OG_whole_training_pdb_ids.append(OG_whole_testing_pdb_ids[12])
OG_whole_training_pdb_ids.append(OG_whole_testing_pdb_ids[4])
OG_whole_training_pdb_ids.append(OG_whole_testing_pdb_ids[0])
#%%
#ONGO_whole_training_unigene.append(ONGO_whole_testing_unigene[3]) 
#ONGO_whole_training_pdb_ids.append(ONGO_whole_testing_pdb_ids[3])
ONGO_whole_testing_unigene.append(ONGO_whole_training_unigene[35]) 
ONGO_whole_testing_pdb_ids.append(ONGO_whole_training_pdb_ids[35])
#%%
#ONGO_missed_pdbs_train, ONGO_missed_pdbs_test = create_data_set("ONGO",ONGO_whole_training_pdb_ids, ONGO_whole_testing_pdb_ids,ONGO_whole_training_unigene, ONGO_whole_testing_unigene)    
#create_data_set("ONGO",ONGO_whole_training_pdb_ids, ONGO_whole_testing_pdb_ids,ONGO_whole_training_unigene, ONGO_whole_testing_unigene)    
create_data_set("ONGO",OG_whole_training_pdb_ids, OG_whole_testing_pdb_ids,OG_whole_training_unigene, OG_whole_testing_unigene)    
#%% then  finaly check the created files satisfy the gene and PDB_details 
def load_created_train_test_file(name):
        saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
        os.chdir("/")
        os.chdir(saving_pik)
        whole_training_unigene =  pickle.load(open( ''.join([name,"_train_uni_gene.p"]), "rb" ))
        whole_training_pdb_ids = pickle.load(open(''.join([name,"_train_pdb_ids.p"]), "rb" ))
        whole_testing_unigene =  pickle.load(open( ''.join([name,"_test_uni_gene.p"]), "rb" ))
        whole_testing_pdb_ids = pickle.load(open(''.join([name,"_test_pdb_ids.p"]), "rb" ))
        return  whole_training_unigene, whole_training_pdb_ids, whole_testing_unigene, whole_testing_pdb_ids
ONGO_whole_training_unigene, ONGO_whole_training_pdb_ids, ONGO_whole_testing_unigene, ONGO_whole_testing_pdb_ids =  load_created_train_test_file("ONGO")
TSG_whole_training_unigene, TSG_whole_training_pdb_ids, TSG_whole_testing_unigene, TSG_whole_testing_pdb_ids =  load_created_train_test_file("TSG")
Fusion_whole_training_unigene, Fusion_whole_training_pdb_ids, Fusion_whole_testing_unigene, Fusion_whole_testing_pdb_ids =  load_created_train_test_file("Fusion")
ONGO_train_flatten = list(set(sum(ONGO_whole_training_pdb_ids,[])))
ONGO_test_flatten = list(set(sum(ONGO_whole_testing_pdb_ids,[])))
TSG_train_flatten = list(set(sum(TSG_whole_training_pdb_ids,[])))
TSG_test_flatten = list(set(sum(TSG_whole_testing_pdb_ids,[])))
Fusion_train_flatten = list(set(sum(Fusion_whole_training_pdb_ids,[])))
Fusion_test_flatten = list(set(sum(Fusion_whole_testing_pdb_ids,[])))
#%% missed PDB_ids by whole set creation
"""If any PDB_ids missed create them seperately"""
from class_alpha_C_surface_results  import alpha_C_surface_results
name = "TSG"
saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
saving_dir_train =  ''.join([saving_pik, "/train/",name])
saving_dir_test =  ''.join([saving_pik, "/test/",name])
pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
for pdb_name in TSG_missed_pdbs_train:
    train = alpha_C_surface_results(pdb_source_dir,saving_dir_train,pdb_name)
    train.create_results()
for pdb_name in TSG_missed_pdbs_test:
    test = alpha_C_surface_results(pdb_source_dir,saving_dir_test,pdb_name)
    test.create_results()