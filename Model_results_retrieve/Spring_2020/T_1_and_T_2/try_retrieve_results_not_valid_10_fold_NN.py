# -*- coding: utf-8 -*-
"""
Created on %14-May-2020 at 04.43p.m
@author: %A.Nishanth C00294860
"""
import os
import numpy as np
from copy import deepcopy
import pickle
#%%
ear_prob=True
num_amino_acids=17
model_name='best_BIR_k3_to_7'
nn_probability_part=''.join(['C:/Users/nishy/Desktop/Spring_2020/T_1_T_2/before_thresh/',str(num_amino_acids),'_chan/T_1_T_2_results/'])
nn_probability_left_pdb_part=''.join(['C:/Users/nishy/Desktop/Spring_2020/T_1_T_2/before_thresh/',str(num_amino_acids),'_chan/left_PDB_results/'])

nn_probability=''.join([nn_probability_part,model_name])
nn_probability_left_pdb=''.join([nn_probability_left_pdb_part,model_name])
'''Load the results'''
    
     #%%
def prob_fix_func(prob,round_fac=2):
    '''
    This function fix the probability added to one and round up the probability
    '''
    prob = np.round(prob,2)
    if sum(prob)==1:
        return prob
    else:
        all_sum=sum(prob)
        prob[0] = prob[0]/all_sum
        prob[1] = prob[1]/all_sum
        prob[2] = prob[2]/all_sum
        return np.round(prob,2)
'''go through each fold classification and extract the results'''
print("")
print("")
print("")

print("start from 0 and end at 10")
print("")
print("")
test_prob_dic={}
for nn_probability_sel in [nn_probability,nn_probability_left_pdb]:
    os.chdir('/')
    os.chdir(nn_probability_sel)
    for nth_fold in range(0,10):
    #nth_fold=1
        test_PDBs= pickle.load(open(''.join(["pdbs_test_",str(nth_fold),".p"]), "rb"))
        test_probabilities = pickle.load(open(''.join(["test_probabilities_",str(nth_fold),".p"]), "rb"))
        #go through the PDBs and retrie the results
        for i in range(0,len(test_PDBs)):
            for j in range(0,len(test_probabilities[i])):
                if nth_fold==0:
                    if i==302 and ear_prob:
                        test_prob_dic[test_PDBs[i][3+j]]=prob_fix_func(test_probabilities[i][j,:])
                    else:
                        test_prob_dic[test_PDBs[i][j]]=prob_fix_func(test_probabilities[i][j,:])

                else:
                    if i==302 and ear_prob:
                        test_prob_dic[test_PDBs[i][3+j]]=test_prob_dic[test_PDBs[i][3+j]]+prob_fix_func(test_probabilities[i][j,:])
                    else:
                        test_prob_dic[test_PDBs[i][j]]=test_prob_dic[test_PDBs[i][j]]+prob_fix_func(test_probabilities[i][j,:])
test_PDBs=[]
for key in test_prob_dic.keys(): 
        test_PDBs.append(key) 
#%%
for pdbs in test_PDBs:
    test_prob_dic[pdbs]=prob_fix_func(test_prob_dic[pdbs]/10)
#%%
def update_or_extend_dic(given_pdb,given_prob,test_prob_dic):
    for i in range(0,len(given_pdb)):
        test_prob_dic[given_pdb[i]]=given_prob[i,:]
    return test_prob_dic

os.chdir('/')
os.chdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Fully_clean/SITE/before_thresh/',str(num_amino_acids),'_amino_acid/Finalised_results/BIR']))

OG_test_probs_numpy = np.load('OG_10_fold_probs.npy')
TSG_test_probs_numpy = np.load('TSG_10_fold_probs.npy')
Fusion_test_probs_numpy =np.load('Fusion_10_fold_probs.npy')

OG_test_pdbs = pickle.load(open("OG_10_fold_PDBs.p", "rb"))  
TSG_test_pdbs = pickle.load(open("TSG_10_fold_PDBs.p", "rb"))  
Fusion_test_pdbs = pickle.load(open("Fusion_10_fold_PDBs.p", "rb"))  
#%%
test_prob_dic=update_or_extend_dic(OG_test_pdbs,OG_test_probs_numpy,test_prob_dic)
test_prob_dic=update_or_extend_dic(TSG_test_pdbs,TSG_test_probs_numpy,test_prob_dic)
test_prob_dic=update_or_extend_dic(Fusion_test_pdbs,Fusion_test_probs_numpy,test_prob_dic)
#%%
test_PDBs=list(set(test_PDBs+OG_test_pdbs+TSG_test_pdbs+Fusion_test_pdbs))

    
#%%
os.chdir('/')    
os.chdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_1/SITE/before_thresh/',str(num_amino_acids),'_aminoacid']))
#OG_V2_pdbs =pickle.load(open("ONGO_satisfied_PDBs.p", "rb"))  

OG_V2_pdbs = list(set(sum(pickle.load(open("ONGO_satisfied_PDBs.p", "rb")),[])))  
TSG_V2_pdbs = list(set(sum(pickle.load(open("TSG_satisfied_PDBs.p", "rb")),[])))  
Fusion_V2_pdbs = list(set(sum(pickle.load(open("Fusion_satisfied_PDBs.p", "rb")),[])))  
 
def given_PDBs_prob(given_pdbs,test_prob_dic): 
    given_pdbs_probs_numpy =np.empty((len(given_pdbs),3))

    for i in range(0,len(given_pdbs)):
#        if given_pdbs[i]!='721P':
#        if given_pdbs[i]!='5AY8':
            chk_pdb=''.join([given_pdbs[i],'.npy'])
            given_pdbs_probs_numpy[i,:]=deepcopy(test_prob_dic[chk_pdb])
    return given_pdbs_probs_numpy
         
ONGO_test_probs_numpy=given_PDBs_prob(OG_V2_pdbs,test_prob_dic)
TSG_test_probs_numpy =given_PDBs_prob(TSG_V2_pdbs,test_prob_dic)
Fusion_test_probs_numpy =given_PDBs_prob(Fusion_V2_pdbs,test_prob_dic)

#pickle.dump(ONGO_test_probs_numpy, open("T1_all_ONGO_probs_numpy.p", "wb"))  
#pickle.dump(TSG_test_probs_numpy, open("T1_all_TSG_probs_numpy.p", "wb"))  
#pickle.dump(Fusion_test_probs_numpy, open("T1_all_Fusion_probs_numpy.p", "wb"))  

#%% old probemetric PDB find
#os.chdir('/')
#os.chdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Fully_clean/SITE/before_thresh/',str(num_amino_acids),'_amino_acid']))


##for organise the results or performance
#train_labels_dic= pickle.load(open("train_labels_dic.p", "rb"))
#overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
#
#    #%%
#os.chdir('/')    
#os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_1/SITE/before_thresh/21_aminoacid')
#
#
#ONGO_satisfied_PDBs=pickle.load(open("ONGO_satisfied_PDBs.p", "rb"))  
##%%
#os.chdir('/')    
#os.chdir('F:/scrach_cacs_1083/optimaly_tilted_17_quarter/10_fold_pikles')
#OG_train_PDBs=pickle.load(open("OG_train_PDBs.p", "rb"))  
#if '721P.npy' in  OG_train_PDBs:
#    print("There") 
#    
#    #%%
#os.chdir('/')    
#os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Fully_clean/SITE/before_thresh/21_amino_acid')
##OG_train_PDBs=pickle.load(open("OG_train_PDBs.p", "rb"))  
##OG_train_PDBs=list(set(sum(pickle.load(open("10_splited_clean_ONGO.p", "rb")),[])))  
##if '721P.npy' in  OG_train_PDBs:
##    print("There") 
#Fusion_train_PDBs =pickle.load(open("Fusion_train_PDBs.p", "rb"))
#if '5AY8.npy' in  Fusion_train_PDBs:
#    print("There") 