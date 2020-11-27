# -*- coding: utf-8 -*-
"""
Created on %03-Feb-2020 at 03.46 P.m

@author: %A.Nishanth C00294860
"""


import os
import numpy as np
import pickle

# make sure the 
from class_retrieve_results_gene_sep_NN  import retrieve_results_gene_sep_NN
from class_ROC_for_given_3_classes_with_probability import ROC_for_given_3_classes_with_probability
from class_probability_weight_cal_results_from_keras_NN  import probability_weight_cal_results_from_keras_NN

#%%
minimum_prob_decide = 0.25 
SITE_MUST=True
old_Thresh=True
'''
here if 20 aminoacids used then use the old property proposed in our previuos paper with SITE must property
Else using the new assigned general chemical properties
'''
num_amino_acids=21
#num_amino_acids=20
"""FOR clean new_ran_clean"""
checking_data='Before_thresh/21_amino_acids/21-prop/'
#checking_data='Before_thresh/20_amino_acids/17-prop/'
model_name="Brain_inception_residual_results/bires_sel"
#model_name="Brain_inception_residual_results/brain_incept_residual_7x1_with_3x3"
#model_name="Brain_inception_only_results"
#model_name="model_par_inception_residual"
#model_name='brain_inc_res/k-3_7'
#model_name='brain_inc_res/brain_incept_k3-7_with_7x1_only'
#model_name='brain_inc_res/brain_incept_res_k3_7x1_3x3'
#model_name='Brain_inception_residual_results/brain_incept_residual_with_7'
#model_name='Brain_inception_residual_results/brain_incept_residual_with_7x1'
#model_name='par_inc_res'
#model_name='brain_inc_only'
#model_name='par_incept_only'
#model_name='model_accidental_use_same_NN_k_7_d_32'

#model_name='par_incept_only'

main_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated/",checking_data])
nn_probability=''.join([main_dir,model_name])
saving_dir_NN_retrieved=''.join([main_dir,'Finalised_results/',model_name])
#%%
'''TO retrieve the results and save as numpy arrays'''
PDB_retrieve_obj=retrieve_results_gene_sep_NN(main_dir,nn_probability,saving_dir_NN_retrieved)
PDB_Accuracies = PDB_retrieve_obj.Accuracies
print(PDB_Accuracies)
#%% To plot ROCs of PDBs
'''
To plot ROCs of PDBs
'''
os.chdir('/')
os.chdir(saving_dir_NN_retrieved)
OG_test_probs_numpy = np.load('OG_test_probs.npy')
TSG_test_probs_numpy = np.load('TSG_test_probs.npy')
Fusion_test_probs_numpy =np.load('Fusion_test_probs.npy')

#OG_test_probs_numpy = np.load('OG_train_probs.npy')
#TSG_test_probs_numpy = np.load('TSG_train_probs.npy')
#Fusion_test_probs_numpy =np.load('Fusion_train_probs.npy')

#OG_test_probs_numpy = np.load('OG_valid_probs.npy')
#TSG_test_probs_numpy = np.load('TSG_valid_probs.npy')
#Fusion_test_probs_numpy =np.load('Fusion_valid_probs.npy')
##
test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
AUC_OG_VS_TSG_pdb=AUC_matrix[1,:]
print('PDB OG: ',AUC_OG_VS_TSG_pdb[0])
print('PDB TSG: ',AUC_OG_VS_TSG_pdb[1])
#%%

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
def gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh(name,loading_NN_prob_dir,Tier,num_amino_acids=21):
    """FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties"""
    if old_Thresh and SITE_MUST:
        if num_amino_acids==21:  
            threshold_dir='old_thresh/21_amino_acids/Clean_no_overlap_at_all_21_amino_prop_with_SITE/'
            unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
            os.chdir('/')
            os.chdir(unigene_load_dir)
            train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
            test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
        
        elif num_amino_acids==20:   
            threshold_dir='old_thresh/20_amino_acids/Clean_no_overlap_at_all_20_amino_prop_with_SITE/'
            unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/20_amino_acids_fully_clean_SITE'
            os.chdir('/')
            os.chdir(unigene_load_dir)
            train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_20.p"]), "rb")) 
            test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_20.p"]), "rb")) 
    working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#    unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean/21_amino_acids_fully_clean"])
    saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/gene_classification_2020/",threshold_dir,model_name])
    
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/Spring_2020/",threshold_dir,name])


    
    train_pdbs,train_probs,test_pdbs,test_probs= load_NN_results_gene_seperated_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name)
    '''train set'''
    obj = probability_weight_cal_results_from_keras_NN(train_new_gene_list,train_pdbs,train_probs, working_dir_fraction,saving_dir_part,name, minimum_prob_decide= 0.25,Test_set=False,straight_save=False,return_out=True)
    train_gene_direct_prob, train_fin_gene_prob, train_Ensamble_gene_prob  = obj.documentation_prob()
    del obj # delete the object to clear the memory
    '''test set'''
    obj = probability_weight_cal_results_from_keras_NN(test_new_gene_list,test_pdbs,test_probs, working_dir_fraction,saving_dir_part,name, minimum_prob_decide= 0.25,Test_set=True,straight_save=False,return_out=True)
    test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob  =    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("clean documentation of ", name," done")
    return train_gene_direct_prob, train_fin_gene_prob, train_Ensamble_gene_prob, test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob

if not (SITE_MUST or old_Thresh):
    print('')
    print('')
    print('Since these weights calculted intially If any data changes like ')
    print('without SITE or ')
    print('New PDBs or ')
    print('Threshold condition ')
    print('May increase or reduce the number of PDBs that define primary gene sequence Functionality So it have to recalculated')
    print('')
    print('Directory "working_dir_fraction" define the weightage of PDBs ')
    print('')
    print('')
    raise("You have to change the weitage in primary gene detection")

Tier=1
"""FOR clean new_ran_clean"""
ONGO_train_gene_direct_prob, ONGO_train_fin_gene_prob, ONGO_train_Ensamble_gene_prob, ONGO_test_gene_direct_prob, ONGO_test_fin_gene_prob, ONGO_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("ONGO",saving_dir_NN_retrieved,Tier,num_amino_acids=num_amino_acids)
TSG_train_gene_direct_prob, TSG_train_fin_gene_prob, TSG_train_Ensamble_gene_prob, TSG_test_gene_direct_prob, TSG_test_fin_gene_prob, TSG_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("TSG",saving_dir_NN_retrieved,Tier,num_amino_acids=num_amino_acids)
Fusion_train_gene_direct_prob, Fusion_train_fin_gene_prob, Fusion_train_Ensamble_gene_prob, Fusion_test_gene_direct_prob, Fusion_test_fin_gene_prob, Fusion_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("Fusion",saving_dir_NN_retrieved,Tier,num_amino_acids=num_amino_acids)
#%%
#test_obj = ROC_for_given_3_classes_with_probability(ONGO_train_Ensamble_gene_prob,TSG_train_Ensamble_gene_prob,Fusion_train_Ensamble_gene_prob,plot_on=True)
#confusion_matrix_test,AUC_matrix  = test_obj.results_final()
#test_obj = ROC_for_given_3_classes_with_probability(ONGO_test_Ensamble_gene_prob,TSG_test_Ensamble_gene_prob,Fusion_test_Ensamble_gene_prob,plot_on=True)
#confusion_matrix_test,AUC_matrix  = test_obj.results_final()
#%% tereve test results of gene
ONGO_all_gene=np.concatenate((ONGO_test_gene_direct_prob,ONGO_test_Ensamble_gene_prob))
TSG_all_gene=np.concatenate((TSG_test_gene_direct_prob,TSG_test_Ensamble_gene_prob))
Fusion_all_gene=np.concatenate((Fusion_test_gene_direct_prob,Fusion_test_Ensamble_gene_prob))
test_obj = ROC_for_given_3_classes_with_probability(ONGO_all_gene,TSG_all_gene,Fusion_all_gene,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
Gene_Acc = test_obj.Accuracy_gene
##%%
#'''checking the training performance'''
#ONGO_all_gene=np.concatenate((ONGO_train_gene_direct_prob,ONGO_train_Ensamble_gene_prob))
#TSG_all_gene=np.concatenate((TSG_train_gene_direct_prob,TSG_train_Ensamble_gene_prob))
#Fusion_all_gene=np.concatenate((Fusion_train_gene_direct_prob,Fusion_train_Ensamble_gene_prob))
#train_obj = ROC_for_given_3_classes_with_probability(ONGO_all_gene,TSG_all_gene,Fusion_all_gene,plot_on=True,bin_conf_mat=True)
#confusion_matrix_train,AUC_matrix,confusion_binary_matrixes  = train_obj.results_final()
#Gene_Acc = train_obj.Accuracy_gene
"""possible I can plot among method 1 vs two vs 3 or form a table like in the paper"""
AUC_OG_VS_TSG_all=AUC_matrix[1,:]
print('All OG: ',AUC_OG_VS_TSG_all[0])
print('All TSG: ',AUC_OG_VS_TSG_all[1])
#%% 
'''Only check ing the Ensemble method gene results'''
test_obj = ROC_for_given_3_classes_with_probability(ONGO_test_Ensamble_gene_prob,TSG_test_Ensamble_gene_prob,Fusion_test_Ensamble_gene_prob,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix_en,confusion_binary_matrixes  = test_obj.results_final()
AUC_OG_VS_TSG_en=AUC_matrix_en[1,:]
print('Ensample OG: ',AUC_OG_VS_TSG_en[0])
print('Ensample TSG: ',AUC_OG_VS_TSG_en[1])
"""possible I can plot among method 1 vs two vs 3 or form a table like in the paper"""
test_obj_method_1 = ROC_for_given_3_classes_with_probability(ONGO_test_fin_gene_prob[0,:,:],TSG_test_fin_gene_prob[0,:,:],Fusion_test_fin_gene_prob[0,:,:],plot_on=False,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix_m1,confusion_binary_matrixes  = test_obj_method_1.results_final()
AUC_OG_VS_TSG_m1=AUC_matrix_m1[1,:]
print('Method_1 OG: ',AUC_OG_VS_TSG_m1[0])
print('Method_1 TSG: ',AUC_OG_VS_TSG_m1[1])

test_obj_method_2 = ROC_for_given_3_classes_with_probability(ONGO_test_fin_gene_prob[1,:,:],TSG_test_fin_gene_prob[1,:,:],Fusion_test_fin_gene_prob[1,:,:],plot_on=False,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix_m2,confusion_binary_matrixes  = test_obj_method_2.results_final()
AUC_OG_VS_TSG_m2=AUC_matrix_m2[1,:]
print('Method_2 OG: ',AUC_OG_VS_TSG_m2[0])
print('Method_2 TSG: ',AUC_OG_VS_TSG_m2[1])

test_obj_method_3 = ROC_for_given_3_classes_with_probability(ONGO_test_fin_gene_prob[2,:,:],TSG_test_fin_gene_prob[2,:,:],Fusion_test_fin_gene_prob[2,:,:],plot_on=False,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix_m3,confusion_binary_matrixes  = test_obj_method_3.results_final()
AUC_OG_VS_TSG_m3=AUC_matrix_m3[1,:]
print('Method_3 OG: ',AUC_OG_VS_TSG_m3[0])
print('Method_3 TSG: ',AUC_OG_VS_TSG_m3[1])