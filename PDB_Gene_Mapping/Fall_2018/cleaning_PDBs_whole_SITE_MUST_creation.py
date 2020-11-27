# -*- coding: utf-8 -*-
"""
Created on %17-November-2018(09.46pm)

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

#% then load the cleaned PDBs and remove the overlapping PDBs and UniGene
def load_clean_gene_pdbs(name):
    """
    load the pikles data of clenaed data information
    """
    
    clean_UniGene = pickle.load(open(''.join([name,"_clean_UniGene.p"]), "rb" ) )
    clean_only_unigene_PDBs = pickle.load(open(''.join([name,"_clean_unigene_PDB.p"]), "rb" ) ) 
    unwanted_PDBs = pickle.load(open(''.join([name,"_overlapped_PDBs.p"]), "rb" ) )         
    return clean_UniGene, clean_only_unigene_PDBs, unwanted_PDBs

uni_gene_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
os.chdir("/")
os.chdir(uni_gene_pdb_dir)
OG_clean_UniGene, OG_clean_only_unigene_PDBs, og_unwanted_PDBs = load_clean_gene_pdbs("ONGO")
TSG_clean_UniGene, TSG_clean_only_unigene_PDBs,tsg_unwanted_PDBs = load_clean_gene_pdbs("TSG")
Fusion_clean_UniGene, Fusion_clean_only_unigene_PDBs,fus_unwanted_PDBs = load_clean_gene_pdbs("Fusion")
#%%
"""

Data creation starts here!!!!!!!!!!!!!!!!!!

"""
"""
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
    whole_training_unigene = pickle.load(open(''.join([name,"_train_uni_gene.p"]), "rb" ) ) 
    # testing data pickle
    whole_testing_unigene = pickle.load(open(''.join([name,"_test_uni_gene.p"]), "rb" ) ) 

#    whole_training_unigene = pickle.load(open(''.join([name,"_whole_train_unigene.p"]), "rb" ) ) 
#    # testing data pickle 
#    whole_testing_unigene = pickle.load(open(''.join([name,"_whole_test_unigene.p"]), "rb" ) ) 
    whole_training_pdb_ids = []
    for u in whole_training_unigene:
        for i in range(0,len(unigene)):
            if unigene[i] == u:
                whole_training_pdb_ids.append(unigene_PDB[i])
    whole_testing_pdb_ids= []
    for u in whole_testing_unigene:
        for i in range(0,len(unigene)):
            if unigene[i] == u:
                whole_testing_pdb_ids.append(unigene_PDB[i])    
    return whole_training_pdb_ids, whole_testing_pdb_ids,whole_training_unigene,whole_testing_unigene
created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/new_random_whole"    
#created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
#created_train_test_list_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/new_ran_whole_2"
OG_whole_training_pdb_ids, OG_whole_testing_pdb_ids,OG_whole_training_unigene, OG_whole_testing_unigene = get_PDB_ids("ONGO",OG_clean_UniGene,OG_clean_only_unigene_PDBs,created_train_test_list_dir)
TSG_whole_training_pdb_ids, TSG_whole_testing_pdb_ids,TSG_whole_training_unigene, TSG_whole_testing_unigene = get_PDB_ids("TSG",TSG_clean_UniGene,TSG_clean_only_unigene_PDBs,created_train_test_list_dir)
Fusion_whole_training_pdb_ids, Fusion_whole_testing_pdb_ids, Fusion_whole_training_unigene, Fusion_whole_testing_unigene = get_PDB_ids("Fusion",Fusion_clean_UniGene,Fusion_clean_only_unigene_PDBs,created_train_test_list_dir)
#%% then check the already created datasets to remove the unwanted PDBs
import shutil

def create_data_set(name,whole_training_pdb_ids,whole_testing_pdb_ids,whole_training_unigene,whole_testing_unigene):
    """
    Inorder to copy the files to copy the results
    
    """
    
    saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
    saving_dir_train =  ''.join([saving_pik, "/train/",name])
    saving_dir_test =  ''.join([saving_pik, "/test_cl/",name])
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
    pickle.dump(whole_testing_unigene, open( ''.join([name,"_test_unoverlap_uni_gene.p"]), "wb" ))
    pickle.dump(whole_testing_pdb_ids, open( ''.join([name,"_test_unoverlap_pdb_ids.p"]), "wb" ) )  
    print(name, " created :)")

#%% creating dataset with unoverlapping PDBs
#create_data_set("ONGO",OG_whole_training_pdb_ids, OG_whole_testing_pdb_ids,OG_whole_training_unigene, OG_whole_testing_unigene)
create_data_set("TSG",TSG_whole_training_pdb_ids, TSG_whole_testing_pdb_ids,TSG_whole_training_unigene, TSG_whole_testing_unigene)
#create_data_set("Fusion",Fusion_whole_training_pdb_ids, Fusion_whole_testing_pdb_ids, Fusion_whole_training_unigene, Fusion_whole_testing_unigene)
#%% creating the test dataset for uncleaned data that is remaining
def create_data_set_uncleaned_test(name,UniGenes,normal_unigenes,normal_unigene_PDBs,test_cleaned_unigene):
    """
    This function will create the dataset for the test set which includes the all PDB ids in the dataset
    """
    saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
    saving_dir_train =  ''.join([saving_pik, "/train_test/",name])
    saving_dir_test =  ''.join([saving_pik, "/test/",name])
    # check whether the apth exists or not and create the directory if not there
    if not os.path.isdir(saving_dir_train):
        os.makedirs(saving_dir_train)      
        print(saving_dir_train, " path created")
    if not os.path.isdir(saving_dir_test):
        os.makedirs(saving_dir_test)      
        print(saving_dir_test, " path created")
    # code starts here
    testing_unigenes = []
    for ug in UniGenes:
        if ug not in test_cleaned_unigene:
            testing_unigenes.append(ug)
    testing_PDB_ids =[]        
    for i in range(0,len(normal_unigenes)):
        if normal_unigenes[i] in testing_unigenes:
            testing_PDB_ids.append(normal_unigene_PDBs[i])
    
    train_test_PDB_ids =[]
    for i in range(0,len(normal_unigenes)):
        if normal_unigenes[i] not in testing_unigenes:
            train_test_PDB_ids.append(normal_unigene_PDBs[i])
            
    test_pdbs = sum(testing_PDB_ids, [])
    train_pdbs = sum(train_test_PDB_ids, [])

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
    pickle.dump(train_test_PDB_ids, open( ''.join([name,"_train_test_PDB_ids.p"]), "wb" ) )  
    pickle.dump(testing_unigenes, open( ''.join([name,"_test_uni_gene.p"]), "wb" ))
    pickle.dump(testing_PDB_ids, open( ''.join([name,"_test_pdb_ids.p"]), "wb" ) )  
    print(name, " created :)")

#%%
create_data_set_uncleaned_test("ONGO",OG_clean_UniGene,OG_unigene,OG_unigene_PDB,OG_whole_testing_unigene)
create_data_set_uncleaned_test("TSG",TSG_clean_UniGene,TSG_unigene,TSG_unigene_PDB,TSG_whole_testing_unigene)
create_data_set_uncleaned_test("Fusion",Fusion_clean_UniGene,Fusion_unigene,Fusion_unigene_PDB,Fusion_whole_testing_unigene)
#%% 
##"""then create the Dataset for any class seperately"""
##  
#traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
#name = "ONGO"
##%% then create the whole set
#OG_whole_testing_unigenes = ["Hs.162233","Hs.365116","Hs.707742","Hs.165950","Hs.338207","Hs.470316","Hs.486502","Hs.507590","Hs.515247","Hs.525622","Hs.591742","Hs.631535","Hs.715871"]
#for i in OG_whole_testing_unigenes:
#    if i not in OG_clean_UniGene:
#        print(i ," not in OG_clean_UniGene")
#
#OG_whole_testing_pdb_ids = []
#for i in OG_whole_testing_unigenes:
#    for u in range(0,len(OG_clean_UniGene)):
#        if OG_clean_UniGene[u] == i:
#            OG_whole_testing_pdb_ids.append(OG_clean_only_unigene_PDBs[u])
##%%then choose the rest of the UniGenes as training
#OG_whole_training_unigenes = []
#for i in OG_clean_UniGene:
#    if i not in OG_whole_testing_unigenes:         
#        OG_whole_training_unigenes.append(i)
#
#OG_whole_training_pdb_ids = []
#for i in OG_whole_training_unigenes:
#    for u in range(0,len(OG_clean_UniGene)):
#        if OG_clean_UniGene[u] == i:
#            OG_whole_training_pdb_ids.append(OG_clean_only_unigene_PDBs[u])
##%%          
#saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
#saving_dir_train =  ''.join([saving_pik, "/train/",name])
#saving_dir_test =  ''.join([saving_pik, "/test/",name])
## check whether the apth exists or not and create the directory if not there
#if not os.path.isdir(saving_dir_train):
#    os.makedirs(saving_dir_train)      
#    print(saving_dir_train, " path created")
#if not os.path.isdir(saving_dir_test):
#    os.makedirs(saving_dir_test)      
#    print(saving_dir_test, " path created")
#        
#train_pdbs = sum(OG_whole_training_pdb_ids, [])
#test_pdbs = sum(OG_whole_testing_pdb_ids, [])
#
## since these surface results is already calculated so copy from that
#source_results_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/",name])
#os.chdir("/")
#os.chdir(source_results_dir)
#missed_pdbs_train =[]
#for pdb_name in train_pdbs:
#    fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
#    if os.path.isfile(fname): 
#        file_xy = ''.join([pdb_name,'_xy.txt'])
#        file_yz = ''.join([pdb_name,'_yz.txt'])
#        file_xz = ''.join([pdb_name,'_xz.txt'])
#        shutil.copy(file_xy, saving_dir_train)          
#        shutil.copy(file_yz, saving_dir_train)    
#        shutil.copy(file_xz, saving_dir_train)
#    else:
#        missed_pdbs_train.append(pdb_name)
#missed_pdbs_test = []
#for pdb_name in test_pdbs:
#    fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
#    if os.path.isfile(fname): 
#        file_xy = ''.join([pdb_name,'_xy.txt'])
#        file_yz = ''.join([pdb_name,'_yz.txt'])
#        file_xz = ''.join([pdb_name,'_xz.txt'])
#        shutil.copy(file_xy, saving_dir_test)          
#        shutil.copy(file_yz, saving_dir_test)    
#        shutil.copy(file_xz, saving_dir_test)
#    else:
#        missed_pdbs_test.append(pdb_name)
##%save the created randomised training unigene pdb ids as pikle files
#os.chdir("/")
#os.chdir(saving_pik)
#pickle.dump(OG_whole_training_unigenes, open( ''.join([name,"_train_uni_gene.p"]), "wb" ))
#pickle.dump(OG_whole_training_pdb_ids, open( ''.join([name,"_train_pdb_ids.p"]), "wb" ) )  
#pickle.dump(OG_whole_testing_unigenes, open( ''.join([name,"_test_unoverlap_uni_gene.p"]), "wb" ))
#pickle.dump(OG_whole_testing_pdb_ids, open( ''.join([name,"_test_unoverlap_pdb_ids.p"]), "wb" ) )  
#print(name, " created :)")
#    #%% 
"""
creating ONGO in different way

"""
#name = "ONGO"
#created_train_test_list_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned/new_whole_2" 
#os.chdir("/")
#os.chdir(created_train_test_list_dir)
##%%
#whole_training_unigene = pickle.load(open(''.join([name,"_train_uni_gene.p"]), "rb" ) ) 
##whole_training_PDBs = pickle.load(open(''.join([name,"_train_pdb_ids.p"]), "rb" ) ) 
## testing data pickle
#whole_testing_unigene = pickle.load(open(''.join([name,"_test_unoverlap_uni_gene.p"]), "rb" ) ) 
##whole_testing_PDBs = pickle.load(open(''.join([name,"_test_unoverlap_pdb_ids.p"]), "rb" ) ) 
##%% removing the ids from the testing set and saving in tthe training set
##get the PDB_ids corresponding to UniGene for testing
#whole_testing_PDBs=[]
#testing_unigene = []
#for ug in whole_testing_unigene:
#    if ug not in OG_clean_UniGene:
#        print(ug)
#    for i in range(0,len(OG_clean_UniGene)):
#        if OG_clean_UniGene[i]==ug:
#            whole_testing_PDBs.append(OG_clean_only_unigene_PDBs[i])
#            testing_unigene.append(ug)
##%%
#swapping_ids =[8,9,10]
#OG_whole_training_unigene = []
#OG_whole_testing_unigene = []
#OG_whole_training_pdb_ids =[]
#OG_whole_testing_pdb_ids =[]
#for i in range(0,len(testing_unigene)):
#    if i not in swapping_ids:
#        OG_whole_testing_pdb_ids.append(whole_testing_PDBs[i])
#        OG_whole_testing_unigene.append(testing_unigene[i])
##%%
#for i in range(0,len(OG_clean_UniGene)):
#    if OG_clean_UniGene[i] not in OG_whole_testing_unigene:
#        OG_whole_training_pdb_ids.append(OG_clean_only_unigene_PDBs[i])
#        OG_whole_training_unigene.append(OG_clean_UniGene[i])           
#
#create_data_set("ONGO",OG_whole_training_pdb_ids, OG_whole_testing_pdb_ids,OG_whole_training_unigene, OG_whole_testing_unigene)
#