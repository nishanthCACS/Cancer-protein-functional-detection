# -*- coding: utf-8 -*-
"""
Created on %%(18-Sep-2018) 17.36Pm

@author: %A.Nishanth C00294860
"""
from class_alpha_C_surface_results  import alpha_C_surface_results
import pickle
import os

#% then choose the dataset and arrange them in the directory for training and testing
traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"

#% When use the SITE info must for training and testing

def load_SITE_must_ids(name):
    os.chdir("/")
    os.chdir(traing_testing_pdb_dir)
    # training data pickle
    training_Uni_Genes= pickle.load(open(''.join([name,"_training_Uni_Genes.p"]), "rb" ) ) 
    training_pdb_ids= pickle.load(open(''.join([name,"_training_pdb_ids.p"]), "rb" ) ) 
    # testing data pickle
    testing_Uni_Genes= pickle.load(open(''.join([name,"_testing_Uni_Genes.p"]), "rb" ) ) 
    testing_pdb_ids= pickle.load(open(''.join([name,"_testing_pdb_ids.p"]), "rb" ) ) 
    
    train_pdbs = sum(training_pdb_ids, [])
    test_pdbs = sum(testing_pdb_ids, [])
    #% intializing the wanted saving directories
    # the dirctory needed to load the pdb files
    pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
#    saving_dir_train =  ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/SITE_must/train"])
#    saving_dir_test =  ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/SITE_must/test"])

#    whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/train"])
#    whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/test"])
#    
#    whole_without_redun_train_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole_without_redun/train"])
#    whole_without_redun_test_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole_without_redun/test"])


    saving_dir_train =  ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/SITE_MUST/train/",name])
    saving_dir_test =  ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/SITE_MUST/test/",name])

    whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/train/",name])
    whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/test/",name])
    
    whole_without_redun_train_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole_without_redun/train/",name])
    whole_without_redun_test_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole_without_redun/test/",name])
    
    count_remain = len(train_pdbs)
    for pdb_name in train_pdbs:
        count_remain = count_remain -1
        #pdb_name = train_pdbs[0]
        train = alpha_C_surface_results(pdb_source_dir,saving_dir_train,pdb_name)
        train.create_results()
        train.copy_pdb_results_to_destination(whole_train_dir)
        train.copy_pdb_results_to_destination(whole_without_redun_train_dir)
        print("Remaining in train class ", name, " and remaining ids: ",count_remain)
        del train
        
    count_remain = len(test_pdbs)    
    for pdb_name in test_pdbs:
        count_remain = count_remain -1
        #pdb_name = test_pdbs[0]
        test = alpha_C_surface_results(pdb_source_dir,saving_dir_test,pdb_name)
        test.create_results()
        test.copy_pdb_results_to_destination(whole_test_dir)
        test.copy_pdb_results_to_destination(whole_without_redun_test_dir)
        print("Remaining in test class ", name, " and remaining ids: ",count_remain)
        del test
    return train_pdbs, testing_pdb_ids, training_Uni_Genes, testing_Uni_Genes

ONGO_train_pdbs, ONGO_testing_pdb_ids, ONGO_training_Uni_Genes, ONGO_testing_Uni_Genes = load_SITE_must_ids("ONGO")
TSG_train_pdbs, TSG_testing_pdb_ids, TSG_training_Uni_Genes, TSG_testing_Uni_Genes = load_SITE_must_ids("TSG")
Fusion_train_pdbs, Fusion_testing_pdb_ids, Fusion_training_Uni_Genes, Fusion_testing_Uni_Genes = load_SITE_must_ids("Fusion")

#% Then add the whole_without_redun pdbs
def load_whole_without_redun_ids(name,train_pdbs, test_pdbs):
    os.chdir("/")
    os.chdir(traing_testing_pdb_dir)
    # training data pickle
    whole_without_redun_training_pdb_ids = pickle.load(open(''.join([name,"_whole_without_redun_training_pdb_ids.p"]), "rb" ) ) 
    # testing data pickle
    whole_without_redun_testing_pdb_ids= pickle.load(open(''.join([name,"_whole_without_redun_testing_pdb_ids.p"]), "rb" ) ) 

    #% intializing the wanted saving directories
    # the dirctory needed to load the pdb files
    pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
#
#    whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/train"])
#    whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/test"])
#    
#    whole_without_redun_train_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole_without_redun/train"])
#    whole_without_redun_test_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole_without_redun/test"])
#    
    
    whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/train/",name])
    whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/test/",name])
    
    whole_without_redun_train_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole_without_redun/train/",name])
    whole_without_redun_test_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole_without_redun/test/",name])
 
    #% then go through the already created training files and attach the left ids results
    not_there = [v for v in whole_without_redun_training_pdb_ids if v not in train_pdbs]  
    count_remain = len(not_there)
    for pdb_name in not_there:
        count_remain = count_remain -1
        #pdb_name = train_pdbs[0]
        train = alpha_C_surface_results(pdb_source_dir,whole_without_redun_train_dir,pdb_name)
        train.create_results()
        train.copy_pdb_results_to_destination(whole_train_dir)
        print("Remaining in train class ", name, " and remaining ids: ",count_remain)
        del train
        
    not_there = [v for v in whole_without_redun_testing_pdb_ids if v not in test_pdbs]  
    count_remain = len(not_there)
    for pdb_name in not_there:
        count_remain = count_remain -1
        #pdb_name = train_pdbs[0]
        test = alpha_C_surface_results(pdb_source_dir,whole_without_redun_test_dir,pdb_name)
        test.create_results()
        test.copy_pdb_results_to_destination(whole_test_dir)
        print("Remaining in test class ", name, " and remaining ids: ",count_remain)
        del test
    return whole_without_redun_training_pdb_ids, whole_without_redun_testing_pdb_ids
ONGO_whole_without_redun_training_pdb_ids, ONGO_whole_without_redun_testing_pdb_ids = load_whole_without_redun_ids("ONGO",ONGO_train_pdbs, ONGO_testing_pdb_ids)
TSG_whole_without_redun_training_pdb_ids, TSG_whole_without_redun_testing_pdb_ids = load_whole_without_redun_ids("TSG",TSG_train_pdbs, TSG_testing_pdb_ids)
Fusion_whole_without_redun_training_pdb_ids, Fusion_whole_without_redun_testing_pdb_ids = load_whole_without_redun_ids("Fusion",Fusion_train_pdbs, Fusion_testing_pdb_ids)
#% then create the whole set
def load_whole_ids(name,train_pdbs, test_pdbs):
    os.chdir("/")
    os.chdir(traing_testing_pdb_dir)
    # training data pickle
    whole_training_pdb_ids = pickle.load(open(''.join([name,"_whole_training_pdb_ids.p"]), "rb" ) ) 
    # testing data pickle
    whole_testing_pdb_ids= pickle.load(open(''.join([name,"_whole_testing_pdb_ids.p"]), "rb" ) ) 

    #% intializing the wanted saving directories
    # the dirctory needed to load the pdb files
    pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])

#    whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/train"])
#    whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/test"])

    whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/train/",name])
    whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/test/",name])
 
    #% then go through the already created training files and attach the left ids results
    not_there = [v for v in whole_training_pdb_ids if v not in train_pdbs]  
    count_remain = len(not_there)
    for pdb_name in not_there:
        count_remain = count_remain -1
        #pdb_name = train_pdbs[0]
        train = alpha_C_surface_results(pdb_source_dir,whole_train_dir,pdb_name)
        train.create_results()
        print("Remaining in train class ", name, " and remaining ids: ",count_remain)
        del train
        
    not_there = [v for v in whole_testing_pdb_ids if v not in test_pdbs]  
    count_remain = len(not_there)
    for pdb_name in not_there:
        count_remain = count_remain -1
        #pdb_name = train_pdbs[0]
        test = alpha_C_surface_results(pdb_source_dir,whole_test_dir,pdb_name)
        test.create_results()
        print("Remaining in test class ", name, " and remaining ids: ",count_remain)
        del test

load_whole_ids("ONGO",ONGO_whole_without_redun_training_pdb_ids, ONGO_whole_without_redun_testing_pdb_ids)
load_whole_ids("TSG",TSG_whole_without_redun_training_pdb_ids, TSG_whole_without_redun_testing_pdb_ids)
load_whole_ids("Fusion",Fusion_whole_without_redun_training_pdb_ids, Fusion_whole_without_redun_testing_pdb_ids)
#%% cheking purpose
#name = "ONGO"
#pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
#saving_dir_train =  ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/SITE_must/train"])
#saving_dir_test =  ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/SITE_must/test"])
#
#whole_train_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/train"])
#whole_test_dir  =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole/test"])
#
#whole_without_redun_train_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole_without_redun/train"])
#whole_without_redun_test_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/",name,"_results/whole_without_redun/test"])
#
#train = alpha_C_surface_results(pdb_source_dir,saving_dir_train,"1S9I")
#train.create_results()
#train.copy_pdb_results_to_destination(whole_train_dir)
#train.copy_pdb_results_to_destination(whole_without_redun_train_dir)
#%% 
"""
creating the random trainig set and test set
"""