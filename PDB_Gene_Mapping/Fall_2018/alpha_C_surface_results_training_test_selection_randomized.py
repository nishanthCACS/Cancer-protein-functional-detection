# -*- coding: utf-8 -*-
"""
Created on %%(21-Oct-2018) 16.13Pm

@author: %A.Nishanth C00294860
"""
#from class_alpha_C_surface_results  import alpha_C_surface_results
import pickle
import os
from random import shuffle
import math
import shutil

def create_random_data_set(saving_pik,name):
    """
    This function create the random data set using UniGene seperated PDB_ids
    """
    #% When use the SITE info must for training and testing
    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
    os.chdir("/")
    os.chdir(site_dir)
    
    #testing_pdb_ids= pickle.load(open(''.join([name,"_testing_pdb_ids.p"]), "rb" ) ) 
    fin_PDB_ids = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))    
    fin_uni_gene = pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))  
#    gene_sat = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]),  "rb"))                               
                        
    #% create random number 
    indexes = list(range(0,len(fin_uni_gene)))
    #then randomize the indexes
    shuffle(indexes)
    #% then choose first 80% of unigene ids as training and rest for testing
    train_uni_gene = []
    train_pdb_ids = []
    for i in range(0,math.ceil(0.8*len(fin_uni_gene))):
        train_uni_gene.append(fin_uni_gene[indexes[i]])
        train_pdb_ids.append(fin_PDB_ids[indexes[i]])
    test_uni_gene = []
    test_pdb_ids = []
    for i in range(math.ceil(0.8*len(fin_uni_gene)),len(fin_uni_gene)):
        test_uni_gene.append(fin_uni_gene[indexes[i]])
        test_pdb_ids.append(fin_PDB_ids[indexes[i]])
    
    saving_dir_train =  ''.join([saving_pik, "/train/",name])
    saving_dir_test =  ''.join([saving_pik, "/test/",name])
    # check whether the apth exists or not and create the directory if not there
    if not os.path.isdir(saving_dir_train):
        os.makedirs(saving_dir_train)      
        print(saving_dir_train, " path created")
    if not os.path.isdir(saving_dir_test):
        os.makedirs(saving_dir_test)      
        print(saving_dir_test, " path created")
        
    train_pdbs = sum(train_pdb_ids, [])
    test_pdbs = sum(test_pdb_ids, [])

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
    pickle.dump(train_uni_gene, open( ''.join([name,"_train_uni_gene.p"]), "wb" ))
    pickle.dump(train_pdb_ids, open( ''.join([name,"_train_pdb_ids.p"]), "wb" ) )  
    pickle.dump(test_uni_gene, open( ''.join([name,"_test_uni_gene.p"]), "wb" ))
    pickle.dump(test_pdb_ids, open( ''.join([name,"_test_pdb_ids.p"]), "wb" ) )  
    print(name, " created :)")
    return missed_pdbs_train, missed_pdbs_test

saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/ran_3"
ONGO_missed_pdbs_train, ONGO_missed_pdbs_test = create_random_data_set(saving_pik, "ONGO")
TSG_missed_pdbs_train, TSG_missed_pdbs_test = create_random_data_set(saving_pik, "TSG")
Fusion_missed_pdbs_train, Fusion_missed_pdbs_test = create_random_data_set(saving_pik, "Fusion")

#create_random_data_set("Fusion")
#%% then create the datasets seperately for the train and test for 
from class_alpha_C_surface_results  import alpha_C_surface_results
name = "TSG"
saving_dir_train =  ''.join([saving_pik, "/train/",name])
saving_dir_test =  ''.join([saving_pik, "/test/",name])
pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
for pdb_name in TSG_missed_pdbs_train:
    train = alpha_C_surface_results(pdb_source_dir,saving_dir_train,pdb_name)
    train.create_results()
    
for pdb_name in TSG_missed_pdbs_test:
    test = alpha_C_surface_results(pdb_source_dir,saving_dir_test,pdb_name)
    test.create_results()
        