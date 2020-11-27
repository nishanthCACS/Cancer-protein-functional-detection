# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import pickle
import os
from copy import deepcopy
import numpy as np
#%%

name='ONGO'
model_name='BIR'

#checking_data = ''.join([clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])
checking_data = 'Fully_clean/SITE/before_thresh/21_amino_acid'
model_name='BIR'
main_dir =''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/",checking_data])
loading_NN_prob_dir = ''.join([main_dir,'/Finalised_results/',model_name])


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
PDB_id,probabilities= load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name)

#%%
    
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_1/SITE/before_thresh/21_aminoacid')
chk_dir_part='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/Fully_clean/SITE/before_thresh/21_amino_acid/'
chk_dir=''.join([chk_dir_part,name])
os.chdir('/')
os.chdir(chk_dir)
selected=[]
files_in_dir=os.listdir(chk_dir)
pdbs_all=[]
for i in range(0,len(files_in_dir)):
    if files_in_dir[i][len(name)+1:len(name)+9]=='method_1':
        selected.append(files_in_dir[i])
        whole= pickle.load(open(files_in_dir[i], "rb"))
        pdb_temp=whole[3]
        pdbs_all.append(deepcopy(pdb_temp))
        
pdbs_all=list(set(sum(pdbs_all,[])))


#%%
#ONGO_method_1_Hs.132966.p
whole= pickle.load(open("ONGO_method_2_Hs.132966.p", "rb"))
pdb_temp = []
sum_length = 0
for j in range(0,len(whole[1])):
    sum_length = sum_length + whole[1][j]
    for k in range(0,len(whole[3])):
        #intially needed to append the fractions 
        if j==0:
            pdb_temp.append(whole[1][j]*whole[2][j][k])
        else:
            pdb_temp[k] = pdb_temp[k] + whole[1][j]*whole[2][j][k]
 
for k in range(0,len(whole[3])):            
    pdb_temp[k] = pdb_temp[k]/sum_length
   #% 
# then gothrough probabilities and caculate the final
p_OG = 0
p_TSG = 0
p_Fusion = 0
count=0
not_satisfied_PDB=True
for k in range(0,len(whole[3])): 
    if pdb_temp[k] > 0:
        for j in range(0,len(PDB_id)):
            if whole[3][k] == PDB_id[j]:
                not_satisfied_PDB=False
                p_OG = p_OG + pdb_temp[k]*probabilities[j][0]
                p_TSG = p_TSG + pdb_temp[k]*probabilities[j][1]
                p_Fusion = p_Fusion + pdb_temp[k]*probabilities[j][2]
                count=count+1
        
if len(whole[3])>count:
    print("Length of the PDBs in the method is ",len(whole[3]))
    print("Taken PDBs for assign probability ",count)
# finalize the probabilities
prob = [p_OG/(p_OG + p_TSG + p_Fusion), p_TSG/(p_OG + p_TSG + p_Fusion),  p_Fusion/(p_OG + p_TSG + p_Fusion)]

#%%