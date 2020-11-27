# -*- coding: utf-8 -*-
"""
Created on %03-Feb-2020 at 10.57 Am
@author: %A.Nishanth C00294860
"""
#%%
import os
import numpy as np
from class_ROC_for_given_3_classes_with_probability import ROC_for_given_3_classes_with_probability


model_name="brain_inception_residual"
main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated/"
#working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_NN_results"])
#unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean"])
loading_dir =''.join([main_dir,'Finalised_results/',model_name])
#%%
os.chdir('/')
os.chdir(loading_dir)
OG_test_probs_numpy = np.load('OG_test_probs.npy')
TSG_test_probs_numpy = np.load('TSG_test_probs.npy')
Fusion_test_probs_numpy =np.load('Fusion_test_probs.npy')

#OG_test_probs_numpy = np.load('OG_train_probs.npy')
#TSG_test_probs_numpy = np.load('TSG_train_probs.npy')
#Fusion_test_probs_numpy =np.load('Fusion_train_probs.npy')

#OG_test_probs_numpy = np.load('OG_valid_probs.npy')
#TSG_test_probs_numpy = np.load('TSG_valid_probs.npy')
#Fusion_test_probs_numpy =np.load('Fusion_valid_probs.npy')
test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,plot_on=False)

#%%
''' 10 fold cross validation set retrieve'''
model_name="Brain_inception_residual_results/reran"
loading_NN_prob_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Finalised_results/',model_name])
os.chdir('/')
os.chdir(loading_NN_prob_dir)
OG_test_probs_numpy= np.load('OG_10_fold_probs.npy')
TSG_test_probs_numpy= np.load('TSG_10_fold_probs.npy')
Fusion_test_probs_numpy= np.load('Fusion_10_fold_probs.npy')
test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy)
confusion_matrix_test,AUC_matrix  = test_obj.results_final()