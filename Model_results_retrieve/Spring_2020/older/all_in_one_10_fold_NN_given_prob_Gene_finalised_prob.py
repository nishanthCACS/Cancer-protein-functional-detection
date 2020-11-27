# -*- coding: utf-8 -*-
"""
Created on %03-Feb-2020 at 03.46 P.m

@author: %A.Nishanth C00294860
"""

import os
import numpy as np
import pickle

# make sure the 
from class_retrieve_results_10_fold_NN  import retrieve_results_10_fold_NN
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


'''For SITE selection'''
clean_must_sat_dir_part_save_parts='Fully_clean/SITE/'
#clean_must_sat_dir_part_save_parts='Fully_clean/W_OUT_SITE/'

"""for threhold"""
threh_dir= 'all_c_alpha_SITE/'
#threh_dir= 'all_c_alpha/'
#threh_dir= 'before_thresh/'
#threh_dir= 'New_thresh/'

if num_amino_acids==21:
    amino_acid_dir='21_amino_acid/'
elif num_amino_acids==20:
    amino_acid_dir='20_amino_acid/'

checking_data = ''.join([clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])
#checking_data='/Before_thresh/21_amino_acids/21-prop/'
#checking_data='/Before_thresh/20_amino_acids/17-prop/'

#checking_data='/Man_thresh/21_amino_acids/21-prop/'
#checking_data='/Man_thresh/20_amino_acids/17-prop/''
model_name='BIR'
#model_name='BIR_7'
#model_name='BIR_7x1'
#model_name='BIR_7x1_with_3x3'

#model_name="Brain_inception_residual_results"
#model_name="Brain_inception_only_results"
#model_name="BIR_quat_with_k3_to_7_vin_1_act_swish_f_inc_1_f_d_1"
#model_name="model_par_inception_residual"
#model_name="model_accidental_p1_3_d1_32"
#model_name="Brain_inception_residual_results/fixed"
#model_name="Brain_inception_residual_results/model_brain_incept_RESIDUAL_quat_with_k3_to_7"
#model_name="Brain_inception_residual_results/bir_with_k3_to_7_with_only_7x1"
#model_name="model_brain_incept_RESIDUAL_quat/BIR_mentor_vin_2_act_swish_f_inc_1_f_d_1"
#model_name="model_par_inception_only_mentor"
#model_name='model_par_inception_residual_mentor_vin_1_quat_flatten_fix_swish_f_inc_1_f_d_1'
#model_name="model_accidental_use_same_NN_p1_3_d1_32"
#model_name="model_accidental_use_same_NN_p1_5_d1_32"
#model_name="model_accidental_use_same_NN_p1_7_d1_32"
#model_name="model_brain_inception_Residual_mentor_vin_2_act_swish_f_inc_1_f_d_1"
#model_name="model_brain_incept_RESIDUAL_quat_with_k3_to_7_with_only_7x1"
#model_name='model_brain_inception_only_mentor_vin_1_act_swish_f_inc_1_f_d_1'
model_name='model_brain_inception_Residual_mentor_vin_1_act_swish_f_inc_1_f_d_1'
main_dir =''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/",checking_data])

nn_probability=''.join([main_dir,model_name])
saving_dir_NN_retrieved = ''.join([main_dir,'Finalised_results/',model_name])

#%%

'''TO retrieve the results and save as numpy arrays'''
PDB_retrieve_obj = retrieve_results_10_fold_NN(main_dir,nn_probability,saving_dir_NN_retrieved)
PDB_Accuracies = PDB_retrieve_obj.Accuracies
print(PDB_Accuracies)
#%% To plot ROCs of PDBs
'''
To plot ROCs of PDBs
'''
os.chdir('/')
os.chdir(saving_dir_NN_retrieved)

OG_test_probs_numpy = np.load('OG_10_fold_probs.npy')
TSG_test_probs_numpy = np.load('TSG_10_fold_probs.npy')
Fusion_test_probs_numpy =np.load('Fusion_10_fold_probs.npy')

test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
AUC_OG_VS_TSG_pdb=AUC_matrix[1,:]
print('PDB OG: ',AUC_OG_VS_TSG_pdb[0])
print('PDB TSG: ',AUC_OG_VS_TSG_pdb[1])
#%%

def load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name):
    '''
    To load the probabilities for gene seperated data
    '''
    os.chdir('/')
    os.chdir(loading_NN_prob_dir)
    if name=="ONGO":
        test_probs_numpy = np.load('OG_10_fold_probs.npy')
        test_pdbs = pickle.load(open("OG_10_fold_PDBs.p", "rb"))  
    else:
        test_probs_numpy = np.load(''.join([name,'_10_fold_probs.npy']))   
        test_pdbs = pickle.load(open(''.join([name,"_10_fold_PDBs.p"]), "rb"))  
    for i in range(0,len(test_pdbs)):
        test_pdbs[i] = test_pdbs[i][0:-4]

    return test_pdbs,test_probs_numpy

#% to load the gene details
def gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh(name,loading_NN_prob_dir,Tier,threshold_dir_sel='man_thresh',num_amino_acids=21):
    
    """
    threshold_dir_sel: should be
                man_thresh/   old_thresh
                
    FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties"""
    if old_Thresh and SITE_MUST:
        if num_amino_acids==21:  
            
            threshold_dir=''.join([threshold_dir_sel,'/21_amino_acids/Clean_no_overlap_at_all_21_amino_prop_with_SITE/'])
            unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
            os.chdir('/')
            os.chdir(unigene_load_dir)
            train_part_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
            test_part_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
        
        elif num_amino_acids==20:   
            threshold_dir=''.join([threshold_dir_sel,'/20_amino_acids/Clean_no_overlap_at_all_20_amino_prop_with_SITE/'])
            unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/20_amino_acids_fully_clean_SITE'
            os.chdir('/')
            os.chdir(unigene_load_dir)
            train_part_gene_list = pickle.load(open(''.join([name,"_train_genes_20.p"]), "rb")) 
            test_part_gene_list = pickle.load(open(''.join([name,"_test_genes_20.p"]), "rb")) 
    test_new_gene_list = train_part_gene_list+ test_part_gene_list

    working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#    unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean/21_amino_acids_fully_clean"])
    saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/10_fold_2020/",threshold_dir,model_name])

    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/Spring_2020/",threshold_dir,name])

 
    
    test_pdbs,test_probs= load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name)
    '''validation set'''
    obj = probability_weight_cal_results_from_keras_NN(test_new_gene_list,test_pdbs,test_probs, working_dir_fraction,saving_dir_part,name, minimum_prob_decide= 0.25,Test_set=True,straight_save=False,return_out=True)
    test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob  =    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("clean documentation of ", name," done")
    return test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob

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
ONGO_test_gene_direct_prob, ONGO_test_fin_gene_prob, ONGO_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("ONGO",saving_dir_NN_retrieved,Tier,num_amino_acids=num_amino_acids)
TSG_test_gene_direct_prob, TSG_test_fin_gene_prob, TSG_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("TSG",saving_dir_NN_retrieved,Tier,num_amino_acids=num_amino_acids)
Fusion_test_gene_direct_prob, Fusion_test_fin_gene_prob, Fusion_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_21_SITE_MUST_old_threh("Fusion",saving_dir_NN_retrieved,Tier,num_amino_acids=num_amino_acids)
#%%
'''checkinga all the genes predictions from the Ensemble method+ Direct gene results'''
ONGO_all_gene=np.concatenate((ONGO_test_gene_direct_prob,ONGO_test_Ensamble_gene_prob))
TSG_all_gene=np.concatenate((TSG_test_gene_direct_prob,TSG_test_Ensamble_gene_prob))
Fusion_all_gene=np.concatenate((Fusion_test_gene_direct_prob,Fusion_test_Ensamble_gene_prob))
test_obj = ROC_for_given_3_classes_with_probability(ONGO_all_gene,TSG_all_gene,Fusion_all_gene,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
Gene_Acc = test_obj.Accuracy_gene
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