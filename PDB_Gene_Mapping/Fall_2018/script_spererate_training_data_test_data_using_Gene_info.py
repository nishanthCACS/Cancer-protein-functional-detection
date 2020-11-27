# -*- coding: utf-8 -*-
"""
Created on %%(17-Sep-2018) 22.44Pm

@author: %A.Nishanth C00294860
"""
from class_spererate_training_data_test_data_using_Gene_info  import spererate_training_data_test_data_using_Gene_info

site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
"BLOCK 1 out of 3"
name = "ONGO"
ONGO = spererate_training_data_test_data_using_Gene_info(site_dir,saving_dir,name)
ONGO.train_test_pikle()
#% creating the model for TSG
"BLOCK 2 out of 3"
name = "TSG"
TSG = spererate_training_data_test_data_using_Gene_info(site_dir,saving_dir,name)
TSG.train_test_pikle()


#% creating the model for Fusion
"BLOCK 3 out of 3"
name = "Fusion"
Fusion =  spererate_training_data_test_data_using_Gene_info(site_dir,saving_dir,name)
Fusion.train_test_pikle()
#%% then choose the dataset and arrange them in the directory for training and testing
import pickle
import os
import shutil
traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
os.chdir("/")
os.chdir(traing_testing_pdb_dir)
#%% When use the SITE info must for training and testing 
#def load_SITE_must_ids(name):

name = "ONGO"
# training data pickle
training_Uni_Genes= pickle.load(open(''.join([name,"_training_Uni_Genes.p"]), "rb" ) ) 
training_pdb_ids= pickle.load(open(''.join([name,"_training_pdb_ids.p"]), "rb" ) ) 
# testing data pickle
testing_Uni_Genes= pickle.load(open(''.join([name,"_testing_Uni_Genes.p"]), "rb" ) ) 
testing_pdb_ids= pickle.load(open(''.join([name,"_testing_pdb_ids.p"]), "rb" ) ) 
#%%
# this loading_R_results_dir contain results file has information without consider
loading_R_results_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results"])
loading_pdb_Source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
SITE_destination_dir_train =   ''.join([loading_R_results_dir, "/SITE_must/train"])
SITE_destination_dir_test = ''.join([loading_R_results_dir, "/SITE_must/test"])
#%%
os.chdir("/")
os.chdir(loading_dir)
files = os.listdir(loading_dir)
# copying the PDB_id
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copy(full_file_name, dest)
    
#%% 
os.chdir("/")
os.chdir(loading_dir)
shutil.copy(files[0], SITE_destination_dir_train)