# -*- coding: utf-8 -*-
"""
Created on %09-July-2020 at 10.41 A.m

@author: %A.Nishanth C00294860

"""
import os
import pickle
import numpy as np

from class_retrieve_results_not_valid_10_fold_NN  import load_comp_results_not_valid_10_fold_NN
from class_retrieve_results_not_valid_10_fold_NN  import retrieve_comp_results_not_valid_10_fold_NN
from class_retrieve_results_10_fold_NN  import retrieve_results_10_fold_NN
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

#%%
os.chdir('/')
os.chdir(nn_probability_left_pdb_part)
pdb_obj_not_valid_10_fold= load_comp_results_not_valid_10_fold_NN(model_name,nn_probability,nn_probability_left_pdb_part,saving_dir_NN_retrieved)
test_prob_dic_not_10_valid = pdb_obj_not_valid_10_fold.test_prob_dic
test_PDBs_not_10_valid = pdb_obj_not_valid_10_fold.test_PDBs
#%% 
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
def find_results_need_PDBS(list_chk,overlap_list,list_1):
    sel_pdb_list=[]
    for pdb in list_chk:
        if pdb[0:-4] in list_1:
            if pdb[0:-4] not in overlap_list:
                sel_pdb_list.append(pdb)
    return sel_pdb_list


ONGO_sel_pdb_list=find_results_need_PDBS(test_PDBs_not_10_valid,overalpped_ONGO,ONGO_SITE_sat_pdbs)
Fusion_sel_pdb_list=find_results_need_PDBS(test_PDBs_not_10_valid,overalpped_Fusion,Fusion_SITE_sat_pdbs)
TSG_sel_pdb_list=find_results_need_PDBS(test_PDBs_not_10_valid,overalpped_TSG,TSG_SITE_sat_pdbs)

def perf_chk_way_1(test_prob_dic_not_10_valid,ONGO_sel_pdb_list,TSG_sel_pdb_list,Fusion_sel_pdb_list,ONGO_SITE_sat_pdbs_t_1,TSG_SITE_sat_pdbs_t_1,Fusion_SITE_sat_pdbs_t_1):
    '''
    This function will finalise the results and present that PDBs not appeared in V88 training process at all
    '''
    
    #% calculte the performance or accuracy of the seledect PDBs calssififed from the new census
    ONGO_sel_prob=np.empty((len(ONGO_sel_pdb_list),3))
    for i in range(0,len(ONGO_sel_pdb_list)):
        ONGO_sel_prob[i,:]=test_prob_dic_not_10_valid[ONGO_sel_pdb_list[i]]
        if ONGO_sel_pdb_list[i][0:-4] in ONGO_SITE_sat_pdbs_t_1:
            print('ONGO ',ONGO_sel_pdb_list[i][0:-4]," 's probability : ",test_prob_dic_not_10_valid[ONGO_sel_pdb_list[i]])
    TSG_sel_prob=np.empty((len(TSG_sel_pdb_list),3))
    for i in range(0,len(TSG_sel_pdb_list)):
        TSG_sel_prob[i,:]=test_prob_dic_not_10_valid[TSG_sel_pdb_list[i]]
        if TSG_sel_pdb_list[i][0:-4] in TSG_SITE_sat_pdbs_t_1:
            print('TSG ',TSG_sel_pdb_list[i][0:-4]," 's probability : ",test_prob_dic_not_10_valid[TSG_sel_pdb_list[i]])       
    Fusion_sel_prob=np.empty((len(Fusion_sel_pdb_list),3))
    for i in range(0,len(Fusion_sel_pdb_list)):
        Fusion_sel_prob[i,:]=test_prob_dic_not_10_valid[Fusion_sel_pdb_list[i]]
        if Fusion_sel_pdb_list[i][0:-4] in Fusion_SITE_sat_pdbs_t_1:
            print('Fusion ',Fusion_sel_pdb_list[i][0:-4]," 's probability : ",test_prob_dic_not_10_valid[Fusion_sel_pdb_list[i]])

    print("ONGO mean accuracy: ",np.mean(ONGO_sel_prob,axis=0),' with ',len(ONGO_sel_prob),' pdbs')
    print("TSG mean accuracy: ",np.mean(TSG_sel_prob,axis=0),' with ',len(TSG_sel_prob),' pdbs')
    print("Fusion mean accuracy: ",np.mean(Fusion_sel_prob,axis=0),' with ',len(Fusion_sel_prob),' pdbs')
    return ONGO_sel_prob,TSG_sel_prob,Fusion_sel_prob

ONGO_sel_prob_w_1 ,TSG_sel_prob_w_1 ,Fusion_sel_prob_w_1 =perf_chk_way_1(test_prob_dic_not_10_valid,ONGO_sel_pdb_list,TSG_sel_pdb_list,Fusion_sel_pdb_list,ONGO_SITE_sat_pdbs_t_1,TSG_SITE_sat_pdbs_t_1,Fusion_SITE_sat_pdbs_t_1)

#%% just take only Tier_2 and check it performance

def find_results_need_PDBS_way_2(overlap_list,list_chk):
    sel_pdb_list=[]
    for pdb in list_chk:
        if pdb not in overlap_list:
            sel_pdb_list.append(''.join([pdb,'.npy']))
    return sel_pdb_list

ONGO_sel_pdb_list=find_results_need_PDBS_way_2(overalpped_ONGO,ONGO_SITE_sat_pdbs_t_2)
TSG_sel_pdb_list=find_results_need_PDBS_way_2(overalpped_TSG,TSG_SITE_sat_pdbs_t_2)
Fusion_sel_pdb_list=find_results_need_PDBS_way_2(overalpped_Fusion,Fusion_SITE_sat_pdbs_t_2)

def perf_chk_way_2(test_prob_dic_not_10_valid,ONGO_sel_pdb_list,TSG_sel_pdb_list,Fusion_sel_pdb_list):
    '''
    This function will finalise the results and present that PDBs not appeared in V88 training process at all
    
    Since the Fusion class has issue in preprocessing those PDBs left
    Thus 
    not_satisfied_model
    is only check for that class
    '''
    print("PERFORMANCE CHECK OF WAY-2")
    SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
    os.chdir('/')
    os.chdir(SITE_dir)

    not_satisfied_model=pickle.load(open('not_satisfied_model.p', "rb"))  #the PDB data unable to convert to supported quarter model formation 
    #% calculte the performance or accuracy of the seledect PDBs calssififed from the new census
    ONGO_sel_prob=np.empty((len(ONGO_sel_pdb_list),3))
    for i in range(0,len(ONGO_sel_pdb_list)):
        ONGO_sel_prob[i,:]=test_prob_dic[ONGO_sel_pdb_list[i]]
        if ONGO_sel_pdb_list[i][0:-4] in ONGO_SITE_sat_pdbs_t_1:
            print('ONGO ',ONGO_sel_pdb_list[i][0:-4]," 's probability : ",test_prob_dic[ONGO_sel_pdb_list[i]])
    TSG_sel_prob=np.empty((len(TSG_sel_pdb_list),3))
    for i in range(0,len(TSG_sel_pdb_list)):
        TSG_sel_prob[i,:]=test_prob_dic[TSG_sel_pdb_list[i]]
        if TSG_sel_pdb_list[i][0:-4] in TSG_SITE_sat_pdbs_t_1:
            print('TSG ',TSG_sel_pdb_list[i][0:-4]," 's probability : ",test_prob_dic[TSG_sel_pdb_list[i]])       
    Fusion_sel_fin_prob=[]
    for i in range(0,len(Fusion_sel_pdb_list)):
        if Fusion_sel_pdb_list[i][0:-4] in not_satisfied_model:
            print('In Fusion class ',Fusion_sel_pdb_list[i][0:-4] , " skipped due to preprocessing issue")
        else:
            Fusion_sel_fin_prob.append(test_prob_dic[Fusion_sel_pdb_list[i]])
        if Fusion_sel_pdb_list[i][0:-4] in Fusion_SITE_sat_pdbs_t_1:
            print('Fusion ',Fusion_sel_pdb_list[i][0:-4]," 's probability : ",test_prob_dic[Fusion_sel_pdb_list[i]])
    
    Fusion_sel_prob=np.empty((len(Fusion_sel_fin_prob),3))
    for i in range(0,len(Fusion_sel_fin_prob)):
        Fusion_sel_prob[i,:]=Fusion_sel_fin_prob[i]

            
    print("ONGO mean accuracy: ",np.mean(ONGO_sel_prob,axis=0),' with ',len(ONGO_sel_prob),' pdbs')
    print("TSG mean accuracy: ",np.mean(TSG_sel_prob,axis=0),' with ',len(TSG_sel_prob),' pdbs')
    print("Fusion mean accuracy: ",np.mean(Fusion_sel_prob,axis=0),' with ',len(Fusion_sel_prob),' pdbs')
    return ONGO_sel_prob,TSG_sel_prob,Fusion_sel_prob

ONGO_sel_prob_w_2 ,TSG_sel_prob_w_2 ,Fusion_sel_prob_w_2=perf_chk_way_2(test_prob_dic,ONGO_sel_pdb_list,TSG_sel_pdb_list,Fusion_sel_pdb_list)
#%%

#%%
##%% plotting AUROC to just see the classification confidence
#'''
#To plot the AUROC to find the confidence of the results of Tier-2
#'''
#from class_ROC_for_given_3_classes_with_probability import ROC_for_given_3_classes_with_probability
#
#test_obj = ROC_for_given_3_classes_with_probability(ONGO_sel_prob,TSG_sel_prob,Fusion_sel_prob,plot_on=True,bin_conf_mat=True)
#confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
#AUC_OG_VS_TSG_pdb=AUC_matrix[1,:]
#print('PDB OG V91: ',AUC_OG_VS_TSG_pdb[0])
#print('PDB TSG V91: ',AUC_OG_VS_TSG_pdb[1])
#
#os.chdir('/')
#os.chdir('C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/2020_SITE_sat')
#amino_acid_6DEC=pickle.load(open("amino_acid_6DEC.p", "rb"))

#%%
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
            final[j,:]=np.round(np.mean(method_1_prob,axis=0),decimals=3)
        elif method==2:
#            print(name,' method_2 average: ',np.mean(method_2_prob,axis=0))
            final[j,:]=np.round(np.mean(method_2_prob,axis=0),decimals=3)
        elif method==3:
#            print(name,' method_3 average: ',np.mean(method_3_prob,axis=0))
            final[j,:]=np.round(np.mean(method_3_prob,axis=0),decimals=3)
        else:
#            print(name,' Ensamble average: ',np.mean(ensamble,axis=0))
            final[j,:]=np.round(np.mean(ensamble,axis=0),decimals=3)

    return final

method_1=get_primary_struc_more_PDB_per_method_wise(1)
method_2=get_primary_struc_more_PDB_per_method_wise(2)
method_3=get_primary_struc_more_PDB_per_method_wise(3)
Ensample=get_primary_struc_more_PDB_per_method_wise(4)
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

name_list=['ONGO','TSG','Fusion']

direct_prob_mean=np.empty((3,3))
ens_prob_mean=np.empty((3,3))
for j in range(0,3):
    name=name_list[j]
    direct_prob_mean[j,:]=np.round(np.mean(direct_all[name],axis=0),decimals=3)
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