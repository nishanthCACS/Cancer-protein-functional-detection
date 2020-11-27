# -*- coding: utf-8 -*-
"""
Created on %19-Nov-2020 at 4.44 p.m

@author: %A.Nishanth C00294860
"""
import os
import numpy as np

os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Fully_clean/SITE/before_thresh/21_amino_acid/Finalised_results/Paper_best_BIR_k3_to_5')
ONGO_prob=np.load('OG_test_probs_numpy.npy')
TSG_prob=np.load('TSG_test_probs_numpy.npy')
Fusion_prob=np.load('Fusion_test_probs_numpy.npy')
#%%
def true_positive_rate(prob,sel,thrsh=0.5):
    '''
    prob: given the probability of the PDBs
    sel: specify which class it belongs
    thrsh: threhold condition to assign as postive class
    '''          
    tp=0
    for i in range(0, len(prob)):
        if prob[i][sel]>=thrsh:
            tp=tp+1
            
        
    return tp/len(prob)

def false_positive_rate(prob_1,prob_2,sel,thrsh=0.5):
    '''
    prob: given the probability of the PDBs
    sel: specify which class it belongs
    thrsh: threhold condition to assign as postive class
    '''          
    fp=0
    for i in range(0, len(prob_1)):
        if prob_1[i][sel]>=thrsh:
            fp=fp+1
            
    for i in range(0, len(prob_2)):
        if prob_2[i][sel]>=thrsh:
            fp=fp+1
            
            
    return fp/(len(prob_1)+len(prob_2))

thrsh=0.971
print("ONGO")
print("TPR: ", true_positive_rate(ONGO_prob,0,thrsh))
print("FPR: ",false_positive_rate(TSG_prob,Fusion_prob,0,thrsh))
#%%
print("TSG")

thrsh=0.89
print("TPR: ", true_positive_rate(TSG_prob,1,thrsh))
print("FPR: ",false_positive_rate(ONGO_prob,Fusion_prob,1,thrsh))
#%%
print("Fusion")

thrsh=0.731
print("TPR: ", true_positive_rate(Fusion_prob,2,thrsh))
print("FPR: ",false_positive_rate(ONGO_prob,TSG_prob,2,thrsh))

