# -*- coding: utf-8 -*-
"""
Created on %09-July-2020 at 10.41 A.m

@author: %A.Nishanth C00294860
editted on 04-Nov-2020 at 12.53 p.m
"""
import os
import pickle
import numpy as np
from class_ROC_for_given_3_classes_with_probability import ROC_for_given_3_classes_with_probability

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
old_Thresh=True
#old_Thresh=False
model_name='best_BIR_k3_to_5'
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
    amino_acid_dir_old='21_amino_acid_old_fusion_overlap/'
elif num_amino_acids==20:
    amino_acid_dir='20_amino_acid/'
    amino_acid_dir_old='20_amino_acid_old/'

threshold_dir_sel_old= ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/",clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir_old])

checking_data = ''.join([clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])
#threshold_dir_sel= ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/Spring_2020/",clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])
threshold_dir_sel= ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/",clean_must_sat_dir_part_save_parts,threh_dir,amino_acid_dir])

main_dir =''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/",checking_data])
nn_probability_main=''.join([main_dir,model_name])
nn_probability=''.join([main_dir,'T_1_T_2/T_1_T_2_results/',model_name])
nn_probability_left_pdb_part=''.join([main_dir,'T_1_T_2/left_PDB_results/',model_name])

os.chdir('/')
if not os.path.exists(nn_probability):
   os.makedirs(nn_probability)
os.chdir(nn_probability)
print("Place the models' T1_T2 results here")
#%%
os.chdir('/')
if not os.path.exists(nn_probability_left_pdb_part):
   os.makedirs(nn_probability_left_pdb_part)
os.chdir(nn_probability_left_pdb_part)
print("Place the models' left PDB results here")

saving_dir_NN_retrieved = ''.join([main_dir,'Finalised_results/',model_name])
saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_Finalised_probabilities_gene/2020_k_fold_results/",checking_data,model_name,'/'])
#%
'''
Run the all_in_one_10_fold_NN_given_prob_Gene_finalised_prob.py first
'''

saving_dir_NN_retrieved = ''.join([main_dir,'Finalised_results/',model_name])
os.chdir('/')
os.chdir(saving_dir_NN_retrieved)
test_prob_dic = pickle.load(open("best_BIR_k3_to_5test_prob_dic.p", "rb"))

'''
load the T_2 & T_1 ONGO, TSG, and Fusion classses PDBs and then those PDBs fell in not-validation set whcih not-overlapped at all in among these classes T_1 and T_2
'''
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_2_pdb_pikles_length_gene')
ONGO_SITE_sat_pdbs_t_2 = list(set(sum(pickle.load(open("ONGO_SITE_satisfied.p", "rb")),[]))) 
TSG_SITE_sat_pdbs_t_2 = list(set(sum(pickle.load(open("TSG_SITE_satisfied.p", "rb")),[])))  
Fusion_SITE_sat_pdbs_t_2 = list(set(sum(pickle.load(open("Fusion_SITE_satisfied.p", "rb")),[])))  
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene')
ONGO_SITE_sat_pdbs_t_1 = list(set(sum(pickle.load(open("ONGO_SITE_satisfied.p", "rb")),[]))) 
TSG_SITE_sat_pdbs_t_1 = list(set(sum(pickle.load(open("TSG_SITE_satisfied.p", "rb")),[])))  
Fusion_SITE_sat_pdbs_t_1 = list(set(sum(pickle.load(open("Fusion_SITE_satisfied.p", "rb")),[])))

#%% find the overlap happended in all_ONGO and all_TSG and  all Fusion combined T1 and T2 combined
def find_overlap(list_1,list_2,list_3):
    '''
    This function find the ovrelapping PDBs among the gicven list_1 with list_2+list_3
    '''
    chk_comp=list_2+list_3
    overalpped=[]
    for pdb in list_1:
        if pdb in chk_comp:
            overalpped.append(pdb)
    return overalpped

ONGO_SITE_sat_pdbs=ONGO_SITE_sat_pdbs_t_2+ONGO_SITE_sat_pdbs_t_1
TSG_SITE_sat_pdbs=TSG_SITE_sat_pdbs_t_2+TSG_SITE_sat_pdbs_t_1
Fusion_SITE_sat_pdbs=Fusion_SITE_sat_pdbs_t_2+Fusion_SITE_sat_pdbs_t_1

overalpped_ONGO=find_overlap(ONGO_SITE_sat_pdbs,TSG_SITE_sat_pdbs,Fusion_SITE_sat_pdbs)
overalpped_TSG=find_overlap(TSG_SITE_sat_pdbs,ONGO_SITE_sat_pdbs,Fusion_SITE_sat_pdbs)
overalpped_Fusion=overalpped_ONGO+overalpped_TSG

#%% Fins the results printed PDBs
def find_results_need_PDBS(overlap_list,list_1):
    sel_pdb_list=[]
    for pdb in list_1:
        if pdb not in overlap_list:
            sel_pdb_list.append(pdb)
    return sel_pdb_list


ONGO_sel_pdb_list=find_results_need_PDBS(overalpped_ONGO,ONGO_SITE_sat_pdbs_t_2)
TSG_sel_pdb_list=find_results_need_PDBS(overalpped_TSG,TSG_SITE_sat_pdbs_t_2)
Fusion_sel_pdb_list=find_results_need_PDBS(overalpped_Fusion,Fusion_SITE_sat_pdbs_t_2)

#%% 
'''
Here find the confustion matrix of the PDBs
'''
from copy import deepcopy

def given_PDBs_prob(given_pdbs,limitation_pdbs=['1O04','3N80','6DEC','1ZUM']):
    mini=0
    for pdb in limitation_pdbs:
        if pdb in given_pdbs:
           mini=mini+1 
           print(pdb, " left")
           given_pdbs.remove(pdb)
    given_pdbs_probs_numpy =np.empty((len(given_pdbs),3))    
    for i in range(0,len(given_pdbs)):
        chk_pdb=''.join([given_pdbs[i],'.npy'])
        given_pdbs_probs_numpy[i,:]=deepcopy(test_prob_dic[chk_pdb])  
    return given_pdbs_probs_numpy

OG_test_probs_numpy= given_PDBs_prob(ONGO_sel_pdb_list)
TSG_test_probs_numpy= given_PDBs_prob(TSG_sel_pdb_list)
Fusion_test_probs_numpy= given_PDBs_prob(Fusion_sel_pdb_list)
#%%
test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,plot_on=False,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
AUC_OG_VS_TSG_pdb=AUC_matrix[1,:]
print('PDB OG V91 Tier 2: ',AUC_OG_VS_TSG_pdb[0])
print('PDB TSG V91 Tier 2: ',AUC_OG_VS_TSG_pdb[1])
PDB_Accuracies_T_2_91 = test_obj.confusion_matrix_to_accuracy(confusion_matrix_test)
print(PDB_Accuracies_T_2_91)



#%% just take only Tier_2 
'''
Here onwards only the primary structures are evaluated
so just clear the work space
'''


Tier =2
names_temp_list=os.listdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/pdb_details_length_source_2020_V_91/T_',str(Tier)]))
saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_gene/2020_k_fold_results/Tier_",str(Tier),"/",checking_data,model_name,'/'])
#for names_temp in names_temp_list:
name='ONGO'
os.chdir('/')
os.chdir(saving_dir_primary_seq_part)

def get_primary_struc_more_PDB_per(name):

    Ensamble_gene_prob=pickle.load(open(''.join([name,"_ensamble_gene_name_m_1_2_3_prob.p"]), "rb"))
    fin_gene_prob=pickle.load(open(''.join([name,"_fin_gene_name_m_1_2_3_prob.p"]), "rb"))
    method_1_prob=np.empty((len(fin_gene_prob),3))
    method_2_prob=np.empty((len(fin_gene_prob),3))
    method_3_prob=np.empty((len(fin_gene_prob),3))
    ensamble=np.empty((len(fin_gene_prob),3))
    
    for i in range(0,len(fin_gene_prob)):
        method_1_prob[i,:]=fin_gene_prob[i][1]
        method_2_prob[i,:]=fin_gene_prob[i][2]
        method_3_prob[i,:]=fin_gene_prob[i][3]
        ensamble[i,:]=Ensamble_gene_prob[i][1]
        
        
    print(name,' method_1 average: ',np.mean(method_1_prob,axis=0))
    print(name,' method_2 average: ',np.mean(method_2_prob,axis=0))
    print(name,' method_2 average: ',np.mean(method_3_prob,axis=0))
    
    print(name,' Ensamble average: ',np.mean(ensamble,axis=0))
    print(' ')

get_primary_struc_more_PDB_per('ONGO')
get_primary_struc_more_PDB_per('TSG')
get_primary_struc_more_PDB_per('Fusion')
#%%
def get_primary_struc_more_PDB_per_method_wise(method):
    name_list=['ONGO','TSG','Fusion']

    final=np.empty((3,3))
    for j in range(0,3):
        name=name_list[j]
        Ensamble_gene_prob=pickle.load(open(''.join([name,"_ensamble_gene_name_m_1_2_3_prob.p"]), "rb"))
        fin_gene_prob=pickle.load(open(''.join([name,"_fin_gene_name_m_1_2_3_prob.p"]), "rb"))
        method_1_prob=np.empty((len(fin_gene_prob),3))
        method_2_prob=np.empty((len(fin_gene_prob),3))
        method_3_prob=np.empty((len(fin_gene_prob),3))
        ensamble=np.empty((len(fin_gene_prob),3))
        
        for i in range(0,len(fin_gene_prob)):
            method_1_prob[i,:]=fin_gene_prob[i][1]
            method_2_prob[i,:]=fin_gene_prob[i][2]
            method_3_prob[i,:]=fin_gene_prob[i][3]
            ensamble[i,:]=Ensamble_gene_prob[i][1]
        
        if method==1:
#            print(name,' method_1 average: ',np.mean(method_1_prob,axis=0))
            final[j,:]=np.mean(method_1_prob,axis=0)
        elif method==2:
#            print(name,' method_2 average: ',np.mean(method_2_prob,axis=0))
            final[j,:]=np.mean(method_2_prob,axis=0)
        elif method==3:
#            print(name,' method_3 average: ',np.mean(method_3_prob,axis=0))
            final[j,:]=np.mean(method_3_prob,axis=0)
        else:
#            print(name,' Ensamble average: ',np.mean(ensamble,axis=0))
            final[j,:]=np.mean(ensamble,axis=0)

    return final

method_1=get_primary_struc_more_PDB_per_method_wise(1)
method_2=get_primary_struc_more_PDB_per_method_wise(2)
method_3=get_primary_struc_more_PDB_per_method_wise(3)
Ensample=get_primary_struc_more_PDB_per_method_wise(4)
#%%

name_list=['ONGO','TSG','Fusion']

direct_prob_mean=np.empty((3,3))
for j in range(0,3):
    name=name_list[j]
    print(name)

#%%
'''
Inorder to get the confusion matrix
'''

from class_ROC_for_given_3_classes_with_probability import ROC_for_given_3_classes_with_probability
import os
import numpy as np
import pickle
from copy import deepcopy

Tier =2
names_temp_list=os.listdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/pdb_details_length_source_2020_V_91/T_',str(Tier)]))
saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_gene/2020_k_fold_results/Tier_",str(Tier),"/",checking_data,model_name,'/'])
#for names_temp in names_temp_list:
os.chdir('/')
os.chdir(saving_dir_primary_seq_part)

name_list=['ONGO','TSG','Fusion']
method_1_all={}
method_2_all={}
method_3_all={}
ensamble_all={}
direct_all={}

final=np.empty((3,3))
for j in range(0,3):
    name=name_list[j]
    Ensamble_gene_prob=pickle.load(open(''.join([name,"_ensamble_gene_name_m_1_2_3_prob.p"]), "rb"))
    fin_gene_prob=pickle.load(open(''.join([name,"_fin_gene_name_m_1_2_3_prob.p"]), "rb"))
    method_1_prob=np.empty((len(fin_gene_prob),3))
    method_2_prob=np.empty((len(fin_gene_prob),3))
    method_3_prob=np.empty((len(fin_gene_prob),3))
    ensamble=np.empty((len(fin_gene_prob),3))
    
    for i in range(0,len(fin_gene_prob)):
        method_1_prob[i,:]=fin_gene_prob[i][1]
        method_2_prob[i,:]=fin_gene_prob[i][2]
        method_3_prob[i,:]=fin_gene_prob[i][3]
        ensamble[i,:]=Ensamble_gene_prob[i][1]
   
    direct_prob=pickle.load(open(''.join([name,"_gene_direct_prob.p"]), "rb"))
    direc_prob_np=np.empty((len(direct_prob),3))

    for i in range(0,len(direct_prob)):
        direc_prob_np[i,:]=direct_prob[i][1]
        
    method_1_all[name]=deepcopy(method_1_prob)
    method_2_all[name]=deepcopy(method_2_prob)
    method_3_all[name]=deepcopy(method_3_prob)
    ensamble_all[name]=deepcopy(ensamble)
    direct_all[name]=deepcopy(direc_prob_np)
#%%    
def confusioin_matrix_function(given_method): 
    test_obj = ROC_for_given_3_classes_with_probability(given_method['ONGO'],given_method['TSG'],given_method['Fusion'],plot_on=False)
    confusion_matrix_test,_  = test_obj.results_final()
    del test_obj
    return confusion_matrix_test
confustion_matrix_all={}
confustion_matrix_all['method_1']=confusioin_matrix_function(method_1_all)
confustion_matrix_all['method_2']=confusioin_matrix_function(method_2_all)
confustion_matrix_all['method_3']=confusioin_matrix_function(method_3_all)
confustion_matrix_all['ensamble']=confusioin_matrix_function(ensamble_all)
confustion_matrix_all['direct']=confusioin_matrix_function(direct_all)
#%%
def bin_confusioin_matrix_function(given_method): 
    test_obj = ROC_for_given_3_classes_with_probability(given_method['ONGO'],given_method['TSG'],given_method['Fusion'],plot_on=False,bin_conf_mat=True)
    _,_,confusion_binary_matrixes  = test_obj.results_final()
    del test_obj
    return confusion_binary_matrixes[0]
bin_confustion_matrix_all={}
bin_confustion_matrix_all['method_1']=bin_confusioin_matrix_function(method_1_all)
bin_confustion_matrix_all['method_2']=bin_confusioin_matrix_function(method_2_all)
bin_confustion_matrix_all['method_3']=bin_confusioin_matrix_function(method_3_all)
bin_confustion_matrix_all['ensamble']=bin_confusioin_matrix_function(ensamble_all)
bin_confustion_matrix_all['direct']=bin_confusioin_matrix_function(direct_all)