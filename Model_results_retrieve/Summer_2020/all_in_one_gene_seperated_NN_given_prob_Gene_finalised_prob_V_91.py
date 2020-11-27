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
'''
here if 20 aminoacids used then use the old property proposed in our previuos paper with SITE must property
Else using the new assigned general chemical properties
'''
num_amino_acids=21
#num_amino_acids=20
SITE_MUST=True
slect_threh=True#need to select the trehold for surface atom or consider all the C_alpha atoms
old_Thresh=False

#model_name='BIR'
model_name='BIR_7'
#model_name='BIR_7x1'
#model_name='BIR_7x1_with_3x3'
model_name_given='model_brain_incept_RESIDUAL_quat_with_k3_to_7_vin_1_act_swish_f_inc_1_f_d_1_SITE_msle_RMSprop_72_bal_bat'
#%%
'''For SITE selection'''
if SITE_MUST:
    clean_must_sat_dir_part_save_parts='Fully_clean/SITE/'
    threh_dir= 'all_c_alpha_SITE/'
else:
    clean_must_sat_dir_part_save_parts='Fully_clean/W_OUT_SITE/'
    threh_dir= 'all_c_alpha/'

"""for threhold"""
if slect_threh:
    if old_Thresh:
        threh_dir= 'before_thresh/'
    else:
        threh_dir= 'New_thresh/'

if num_amino_acids==21:
    amino_acid_dir='21_amino_acid/'
elif num_amino_acids==20:
    amino_acid_dir='20_amino_acid/'

checking_data = ''.join([clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])
threshold_dir_sel= ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/",clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])
main_dir =''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated/",checking_data])
nn_probability=''.join([main_dir,model_name])
saving_dir_NN_retrieved=''.join([main_dir,'Finalised_results/',model_name])
os.chdir('/')
if not os.path.exists(nn_probability):
   os.makedirs(nn_probability)
os.chdir(nn_probability)
saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_Finalised_probabilities_gene/2020_gene_seperated/",checking_data,model_name,'/'])
#%%
'''TO retrieve the results and save as numpy arrays'''
PDB_retrieve_obj=retrieve_results_gene_sep_NN(main_dir,nn_probability,saving_dir_NN_retrieved,model_name=model_name_given)
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
def gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh(name,loading_NN_prob_dir,saving_dir_part,threshold_dir_sel,SITE_MUST):
    """FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties"""
    if SITE_MUST:
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
        test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
    '''
    the directly is already placed correctly
    unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'

     just need to do the results retrive for overlaling genes of ONGO and TSG
            pickle.dump(overlapped_genes,open(''.join([name,"_overlapped_genes_21.p"]), "wb")) 
    
    
    '''
    working_dir_fraction = "".join([threshold_dir_sel,name])

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

"""FOR clean new_ran_clean"""
ONGO_train_gene_direct_prob, ONGO_train_fin_gene_prob, ONGO_train_Ensamble_gene_prob, ONGO_test_gene_direct_prob, ONGO_test_fin_gene_prob, ONGO_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("ONGO",saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST)
TSG_train_gene_direct_prob, TSG_train_fin_gene_prob, TSG_train_Ensamble_gene_prob, TSG_test_gene_direct_prob, TSG_test_fin_gene_prob, TSG_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("TSG",saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST)
Fusion_train_gene_direct_prob, Fusion_train_fin_gene_prob, Fusion_train_Ensamble_gene_prob, Fusion_test_gene_direct_prob, Fusion_test_fin_gene_prob, Fusion_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("Fusion",saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST)
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