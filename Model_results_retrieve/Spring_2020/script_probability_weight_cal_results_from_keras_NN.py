# -*- coding: utf-8 -*-
"""
Created on %(02-02-2020) 9.36Am

@author: %A.Nishanth C00294860
"""

import os
import pickle
import numpy as np
# make sure the 
from class_probability_weight_cal_results_from_keras_NN  import probability_weight_cal_results_from_keras_NN
minimum_prob_decide = 0.25 
Tier = 1
#%%
'''
Gene seperated model's results
'''
def load_NN_results_gene_seperated_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name):
    '''
    To load the probabilities for gene seperated data
    '''
    
    os.chdir('/')
    os.chdir(loading_NN_prob_dir)
    if name=="ONGO":
        test_probs= np.load('OG_test_probs.npy')
        valid_probs_numpy = np.load('OG_valid_probs.npy')
        train_probs_numpy = np.load('OG_train_probs.npy')
        
        valid_pdbs = pickle.load(open("OG_valid_PDBs.p", "rb"))  
        train_pdbs_temp= pickle.load(open("OG_train_PDBs.p", "rb"))  
        test_pdbs= pickle.load(open("OG_test_PDBs.p", "rb"))  
    
    else:
        test_probs= np.load(''.join([name,'_test_probs.npy']))
        valid_probs_numpy = np.load(''.join([name,'_valid_probs.npy']))
        train_probs_numpy = np.load(''.join([name,'_train_probs.npy']))
    
        valid_pdbs = pickle.load(open(''.join([name,"_valid_PDBs.p"]), "rb"))  
        train_pdbs_temp= pickle.load(open(''.join([name,"_train_PDBs.p"]), "rb"))  
        test_pdbs= pickle.load(open(''.join([name,"_test_PDBs.p"]), "rb"))  
        
    train_pdbs=[]
    train_probs=[]
    for i in range(0,len(train_pdbs_temp)):
        train_pdbs.append(train_pdbs_temp[i][0:-4])
        train_probs.append(train_probs_numpy[i,:])
    for i in range(0,len(valid_pdbs)):
        train_pdbs.append(valid_pdbs[i][0:-4])
        train_probs.append(valid_probs_numpy[i,:])
    for i in range(0,len(test_pdbs)):
        test_pdbs[i] = test_pdbs[i][0:-4]
    return train_pdbs,train_probs,test_pdbs,test_probs

#% to load the gene details
def gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh(name,loading_NN_prob_dir):
    """FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties"""
    working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#    unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean/21_amino_acids_fully_clean"])
    saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/gene_classification_2020/",model_name])
    
    working_dir_fraction = "".join([working_dir_part, "/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/Spring_2020/Clean_no_overlap_at_all_21_amino_prop_with_SITE/",name])

    unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean'
    os.chdir('/')
    os.chdir(unigene_load_dir)
    train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
    test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
    
    train_pdbs,train_probs,test_pdbs,test_probs= load_NN_results_gene_seperated_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name)
    '''train set'''
    saving_dir =  "".join([saving_dir_part,'/train/',name])
    obj = probability_weight_cal_results_from_keras_NN(train_new_gene_list,train_pdbs,train_probs, working_dir_fraction,saving_dir,name, minimum_prob_decide= 0.25)
    obj.documentation_prob()
    del obj # delete the object to clear the memory
    '''test set'''
    saving_dir =  "".join([saving_dir_part,'/test/',name])
    obj = probability_weight_cal_results_from_keras_NN(test_new_gene_list,test_pdbs,test_probs, working_dir_fraction,saving_dir,name, minimum_prob_decide= 0.25)
    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("clean documentation of ", name," done")

"""FOR clean new_ran_clean"""
model_name="brain_inception_residual"
main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated/"
loading_NN_prob_dir =''.join([main_dir,'Finalised_results/',model_name])
gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("ONGO",loading_NN_prob_dir)
gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("TSG",loading_NN_prob_dir)
gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("Fusion",loading_NN_prob_dir)

#%%
'''
 Here onwards 10-folds set gene classifiaction is done
 specify the model_name
 
'''
def load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name):
    '''
    To load the probabilities for gene seperated data
    '''
    
    os.chdir('/')
    os.chdir(loading_NN_prob_dir)
    if name=="ONGO":
        test_probs= np.load('OG_10_fold_probs.npy')
        test_pdbs= pickle.load(open("OG_10_fold_PDBs.p", "rb"))   
    else:
        test_probs= np.load(''.join([name,'_10_fold_probs.npy']))
        test_pdbs= pickle.load(open(''.join([name,'_10_fold_PDBs.p']), "rb"))
    for i in range(0,len(test_pdbs)):
        test_pdbs[i] = test_pdbs[i][0:-4]
    return test_probs,test_pdbs

def ten_fold_results_finalise_fully_clean_21_SITE_MUST_old_threh(name,loading_NN_prob_dir,model_name):
    """FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties"""
    working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#    unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean/21_amino_acids_fully_clean"])
    saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/10_fold_2020/",model_name]) 
    working_dir_fraction = "".join([working_dir_part, "/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/Spring_2020/Clean_no_overlap_at_all_21_amino_prop_with_SITE/",name])
    unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean'
    os.chdir('/')
    os.chdir(unigene_load_dir)
    train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
#    test_new_gene_list_t = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
    test_probs,test_pdbs= load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name)
    #since the same geneset for 10-fold cross validation as well
    test_new_gene_list=train_new_gene_list#test_new_gene_list_t
    '''10_fold_'s_validation set'''
    saving_dir =  "".join([saving_dir_part,'/',model_name,'/',name])
    obj = probability_weight_cal_results_from_keras_NN(test_new_gene_list,test_pdbs,test_probs, working_dir_fraction,saving_dir,name, minimum_prob_decide= 0.25)
    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("clean documentation of ", name," done")

model_name="Brain_inception_residual_results/reran"
loading_NN_prob_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Finalised_results/',model_name])

ten_fold_results_finalise_fully_clean_21_SITE_MUST_old_threh("ONGO",loading_NN_prob_dir,model_name)
#ten_fold_results_finalise_fully_clean_21_SITE_MUST_old_threh("TSG",loading_NN_prob_dir,model_name)
#ten_fold_results_finalise_fully_clean_21_SITE_MUST_old_threh("Fusion",loading_NN_prob_dir,model_name)


'''Check'''
#%%

chk =pickle.load(open('ONGO_method_1_Hs.195659.p', "rb")) 
#%%
test_probs,test_pdbs= load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,"ONGO")
for pdb in chk[3]:
    if pdb not in test_pdbs:
        print(pdb)
        
#%%
main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results"
#%%
#os.chdir('/')
#os.chdir(main_dir)
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated')
#for organise the results or performance
#OG_train_PDBs=  pickle.load(open("OG_train_PDBs.p", "rb"))
overlapped_PDBs_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
for pdb in chk[3]:
    pdb_t=''.join([pdb,'.npy'])
    if pdb_t in overlapped_PDBs_ONGO:
        print(pdb)
#%%
#name="ONGO"
#model_name="brain_inception_residual"
#main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated/"
#loading_NN_prob_dir =''.join([main_dir,'Finalised_results/',model_name])
#os.chdir('/')
#os.chdir(loading_NN_prob_dir)
#if name=="ONGO":
#    test_probs= np.load('OG_test_probs.npy')
#    valid_probs_numpy = np.load('OG_valid_probs.npy')
#    train_probs_numpy = np.load('OG_train_probs.npy')
#    
#    valid_pdbs = pickle.load(open("OG_valid_PDBs.p", "rb"))  
#    train_pdbs_temp= pickle.load(open("OG_train_PDBs.p", "rb"))  
#    test_pdbs= pickle.load(open("OG_test_PDBs.p", "rb"))  