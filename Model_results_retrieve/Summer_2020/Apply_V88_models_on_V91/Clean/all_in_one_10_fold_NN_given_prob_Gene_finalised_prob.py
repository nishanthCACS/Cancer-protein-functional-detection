# -*- coding: utf-8 -*-
"""
Created on %03-Feb-2020 at 03.46 P.m

@author: %A.Nishanth C00294860

editted on 10-May-2020 at 4.53pm
editted on 27-May-2020 at 11.06p.m

"""

import os
import numpy as np
import pickle

# make sure the 
from class_retrieve_results_not_valid_10_fold_NN  import retrieve_comp_results_not_valid_10_fold_NN
from class_retrieve_results_10_fold_NN  import retrieve_results_10_fold_NN
from class_ROC_for_given_3_classes_with_probability import ROC_for_given_3_classes_with_probability
from class_probability_weight_cal_results_from_keras_NN  import probability_weight_cal_results_from_keras_NN
from class_T_1_T_2_probability_weight_cal_results_from_keras_NN import probability_weight_cal_results_from_keras_NN_for_T_1_T_2
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

#model_name='best_BIR_k3_to_7'
#model_name='best_accidental_same_NN_k_3'
#model_name='best_accidental_same_NN_k_5'
#model_name='best_BIR_k3_to_7_with_only_7x1'
#model_name='best_BIR_k3_to_7_with_7x1_and_3x3'
#model_name='best_BI_only'
#model_name='BIR_7'
#model_name='BIR_7x1'
#model_name='best_BIR_k3_to_7'
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
'''TO retrieve the results and save as numpy arrays'''
PDB_retrieve_obj = retrieve_results_10_fold_NN(main_dir,nn_probability_main,saving_dir_NN_retrieved)
PDB_Accuracies = PDB_retrieve_obj.Accuracies
print(PDB_Accuracies)
#%
'''
To plot ROCs of PDBs
'''
os.chdir('/')
os.chdir(saving_dir_NN_retrieved)

OG_test_probs_numpy = np.load('OG_10_fold_probs.npy')
TSG_test_probs_numpy = np.load('TSG_10_fold_probs.npy')
Fusion_test_probs_numpy =np.load('Fusion_10_fold_probs.npy')
#%%
test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
AUC_OG_VS_TSG_pdb=AUC_matrix[1,:]
print('PDB OG: ',AUC_OG_VS_TSG_pdb[0])
print('PDB TSG: ',AUC_OG_VS_TSG_pdb[1])
'''
get the genes PDB details of ONGO and TSG and Fusion to create the ROC from 2018 census model to 2020 V 91 results
'''
print("")
print(" V91 ")

T_1_T_2_obj = retrieve_comp_results_not_valid_10_fold_NN(model_name,nn_probability,nn_probability_left_pdb_part,saving_dir_NN_retrieved)

Tier=1
if SITE_MUST:
    sat_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_',str(Tier),'/SITE/',threh_dir,str(num_amino_acids),'_aminoacid'])
#OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy= T_1_T_2_obj.Helper_to_get_T_1_ROC_for_differ_version(sat_load_dir)
OG_V2_pdbs,TSG_V2_pdbs,Fusion_V2_pdbs,OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy= T_1_T_2_obj.Helper_to_get_T_1_ROC_for_differ_version(sat_load_dir,PDB_list_icluded=True)

T_1_T_2_obj.update_the_dic_keys()
T1_T2_test_prob_dic= T_1_T_2_obj.test_prob_dic
#%---------------------------
test_obj = ROC_for_given_3_classes_with_probability(OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,plot_on=True,bin_conf_mat=True)
confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
AUC_OG_VS_TSG_pdb=AUC_matrix[1,:]
print('PDB OG V91: ',AUC_OG_VS_TSG_pdb[0])
print('PDB TSG V91: ',AUC_OG_VS_TSG_pdb[1])
PDB_Accuracies_T_1 = test_obj.confusion_matrix_to_accuracy(confusion_matrix_test)
print(PDB_Accuracies_T_1)

#%% 

def load_NN_results_10_fold_fully_clean_SITE_MUST_old_threh(loading_NN_prob_dir,name):
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
def gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh_T_1_T_2(name,T1_T2_test_prob_dic,saving_dir_part,SITE_MUST,num_amino_acids,Tier=1,straight_save=True):
    
    """

    FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties
    """
    if SITE_MUST:
        unigene_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_',str(Tier),'/SITE/before_thresh/',str(num_amino_acids),'_aminoacid'])
        os.chdir('/')
        os.chdir(unigene_load_dir)
        test_new_gene_list = pickle.load(open(''.join([name,"_satisfied_Gene_ID.p"]), "rb")) 
        if name=='TSG' and Tier==1:
            test_new_gene_list.remove('9113')
            print('9113 removed')
    else:
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/W_OUT_SITE'
        os.chdir('/')
        os.chdir(unigene_load_dir)
 
    working_dir_fraction = unigene_load_dir

#    test_pdbs,test_probs= load_NN_results_10_fold_fully_clean_21_SITE_MUST_old_threh(loading_NN_prob_dir,name)
    '''validation set'''
#    obj = probability_weight_cal_results_from_keras_NN(test_new_gene_list,test_pdbs,test_probs, working_dir_fraction,saving_dir_part,name, minimum_prob_decide= 0.25,Test_set=True,straight_save=False,return_out=True)
    obj = probability_weight_cal_results_from_keras_NN_for_T_1_T_2(test_new_gene_list,T1_T2_test_prob_dic, working_dir_fraction,saving_dir_part,name, minimum_prob_decide= 0.25,Test_set=True,straight_save=straight_save,return_out=True)
    test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob  =    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("clean documentation of ", name," done")
    return test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob

def gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh(name,loading_NN_prob_dir,saving_dir_part,threshold_dir_sel,SITE_MUST,num_amino_acids):
    
    """
    threshold_dir_sel: should be
                man_thresh/   old_thresh
                
    FOR clean new_ran_clean and SITE included 21 Amino acids with 21 properties
    """
    if SITE_MUST:
        unigene_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/',str(num_amino_acids),'_amino_acids_fully_clean_SITE'])
        os.chdir('/')
        os.chdir(unigene_load_dir)
        if num_amino_acids==21:
            train_part_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
            test_part_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
        elif num_amino_acids==20:
            train_part_gene_list = pickle.load(open(''.join([name,"_train_genes_20.p"]), "rb")) 
            test_part_gene_list = pickle.load(open(''.join([name,"_test_genes_20.p"]), "rb")) 
    else:
        raise('Not dependent on number of Aminoacids since the number of PDBs is high, so try to include all the PDBs as possible')
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/W_OUT_SITE'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        train_part_gene_list = pickle.load(open(''.join([name,"_train_unigene_details.p"]), "rb")) 
        test_part_gene_list = pickle.load(open(''.join([name,"_test_unigene_details.p"]), "rb")) 
 
    test_new_gene_list = train_part_gene_list+ test_part_gene_list
    working_dir_fraction = "".join([threshold_dir_sel,name])

    test_pdbs,test_probs= load_NN_results_10_fold_fully_clean_SITE_MUST_old_threh(loading_NN_prob_dir,name)
    '''validation set'''
    obj = probability_weight_cal_results_from_keras_NN(test_new_gene_list,test_pdbs,test_probs, working_dir_fraction,saving_dir_part,name, minimum_prob_decide= 0.25,Test_set=True,straight_save=False,return_out=True)
    test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob  =    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("clean documentation of ", name," done")
    print("Go through OLD_done")

    return test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob


def find_ONGO_VS_TSG_VS_Fusion_per_gene(Version_new,saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST,num_amino_acids,feedback=False):
#    """FOR clean new_ran_clean"""
    if Version_new:
        ONGO_test_gene_direct_prob, ONGO_test_fin_gene_prob, ONGO_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh_T_1_T_2("ONGO",T1_T2_test_prob_dic,saving_dir_primary_seq_part,SITE_MUST,num_amino_acids)
        TSG_test_gene_direct_prob, TSG_test_fin_gene_prob, TSG_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh_T_1_T_2("TSG",T1_T2_test_prob_dic,saving_dir_primary_seq_part,SITE_MUST,num_amino_acids)
        Fusion_test_gene_direct_prob, Fusion_test_fin_gene_prob, Fusion_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh_T_1_T_2("Fusion",T1_T2_test_prob_dic,saving_dir_primary_seq_part,SITE_MUST,num_amino_acids)
    else:
        print("Go through OLD")
        ONGO_test_gene_direct_prob, ONGO_test_fin_gene_prob, ONGO_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh("ONGO",saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST,num_amino_acids)
        TSG_test_gene_direct_prob, TSG_test_fin_gene_prob, TSG_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh("TSG",saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST,num_amino_acids)
        Fusion_test_gene_direct_prob, Fusion_test_fin_gene_prob, Fusion_test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh("Fusion",saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST,num_amino_acids)

    '''checkinga all the genes predictions from the Ensemble method+ Direct gene results'''
    ONGO_all_gene=np.concatenate((ONGO_test_gene_direct_prob,ONGO_test_Ensamble_gene_prob))
    TSG_all_gene=np.concatenate((TSG_test_gene_direct_prob,TSG_test_Ensamble_gene_prob))
    Fusion_all_gene=np.concatenate((Fusion_test_gene_direct_prob,Fusion_test_Ensamble_gene_prob))
    test_obj = ROC_for_given_3_classes_with_probability(ONGO_all_gene,TSG_all_gene,Fusion_all_gene,plot_on=False,bin_conf_mat=True)
    test_obj.feedback=feedback
    confusion_matrix_test,AUC_matrix,confusion_binary_matrixes  = test_obj.results_final()
    Gene_Acc = test_obj.Accuracy_gene
#    AUC_OG_VS_TSG_all=AUC_matrix[1,:]
    #print('All OG: ',AUC_OG_VS_TSG_all[0])
    #print('All TSG: ',AUC_OG_VS_TSG_all[1])
    #% 
    '''Only check ing the Ensemble method gene results'''
    test_obj = ROC_for_given_3_classes_with_probability(ONGO_test_Ensamble_gene_prob,TSG_test_Ensamble_gene_prob,Fusion_test_Ensamble_gene_prob,plot_on=False,bin_conf_mat=True)
    test_obj.feedback=feedback
    _,AUC_matrix_en,_  = test_obj.results_final()
#    AUC_OG_VS_TSG_en=AUC_matrix_en[1,:]
    #print('Ensample OG: ',AUC_OG_VS_TSG_en[0])
    #print('Ensample TSG: ',AUC_OG_VS_TSG_en[1])
    """possible I can plot among method 1 vs two vs 3 or form a table like in the paper"""
    test_obj_method_1 = ROC_for_given_3_classes_with_probability(ONGO_test_fin_gene_prob[0,:,:],TSG_test_fin_gene_prob[0,:,:],Fusion_test_fin_gene_prob[0,:,:],plot_on=False,bin_conf_mat=True)
    test_obj_method_1.feedback=feedback
    _,AUC_matrix_m1,_  = test_obj_method_1.results_final()
    AUC_OG_VS_TSG_m1=AUC_matrix_m1[1,:]
    #print('Method_1 OG: ',AUC_OG_VS_TSG_m1[0])
    #print('Method_1 TSG: ',AUC_OG_VS_TSG_m1[1])
    
    test_obj_method_2 = ROC_for_given_3_classes_with_probability(ONGO_test_fin_gene_prob[1,:,:],TSG_test_fin_gene_prob[1,:,:],Fusion_test_fin_gene_prob[1,:,:],plot_on=False,bin_conf_mat=True)
    test_obj_method_2.feedback=feedback
    confusion_matrix_test,AUC_matrix_m2,confusion_binary_matrixes  = test_obj_method_2.results_final()
    AUC_OG_VS_TSG_m2=AUC_matrix_m2[1,:]
    #print('Method_2 OG: ',AUC_OG_VS_TSG_m2[0])
    #print('Method_2 TSG: ',AUC_OG_VS_TSG_m2[1])
    
    test_obj_method_3 = ROC_for_given_3_classes_with_probability(ONGO_test_fin_gene_prob[2,:,:],TSG_test_fin_gene_prob[2,:,:],Fusion_test_fin_gene_prob[2,:,:],plot_on=False,bin_conf_mat=True)
    test_obj_method_3.feedback=feedback
    _,AUC_matrix_m3,_  = test_obj_method_3.results_final()
    AUC_OG_VS_TSG_m3=AUC_matrix_m3[1,:]
    #print('Method_3 OG: ',AUC_OG_VS_TSG_m3[0])
    #print('Method_3 TSG: ',AUC_OG_VS_TSG_m3[1])
    return Gene_Acc,AUC_matrix_en,AUC_OG_VS_TSG_m1,AUC_OG_VS_TSG_m2,AUC_OG_VS_TSG_m3
Version_new=True
Gene_Acc_T1,T1_AUC_matrix_en,T1_AUC_OG_VS_TSG_m1,T1_AUC_OG_VS_TSG_m2,T1_AUC_OG_VS_TSG_m3=find_ONGO_VS_TSG_VS_Fusion_per_gene(Version_new,saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST,num_amino_acids,feedback=False)
Version_new=False
Gene_Acc,AUC_matrix_en,AUC_OG_VS_TSG_m1,AUC_OG_VS_TSG_m2,AUC_OG_VS_TSG_m3=find_ONGO_VS_TSG_VS_Fusion_per_gene(Version_new,saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel,SITE_MUST,num_amino_acids,feedback=False)
Gene_Acc_old,AUC_matrix_en_old,old_AUC_OG_VS_TSG_m1,old_AUC_OG_VS_TSG_m2,old_AUC_OG_VS_TSG_m3=find_ONGO_VS_TSG_VS_Fusion_per_gene(Version_new,saving_dir_NN_retrieved,saving_dir_primary_seq_part,threshold_dir_sel_old,SITE_MUST,num_amino_acids,feedback=False)

#%%
#'''
#This part to create the results for Ti1r 1 and Tier 2 of V91
#'''
##% Gothrough the T_1 and create the results
#for Tier in [1,2]:
#    names_temp_list=os.listdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/pdb_details_length_source_2020_V_91/T_',str(Tier)]))
#    saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_gene/2020_k_fold_results/Tier_",str(Tier),"/",checking_data,model_name,'/'])
#    for names_temp in names_temp_list:
#        name=names_temp[4:-4]
#        if name=='ONGO_TSG' and Tier==1:
#            print("Skipped due to overlapping PDB the results of Tier: ",Tier, " of amino_acids: ",num_amino_acids," name ",name)
#        else:
#            print("creating the results of Tier: ",Tier, " of amino_acids: ",num_amino_acids," name ",name)
#            test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh_T_1_T_2(name,T1_T2_test_prob_dic,saving_dir_primary_seq_part,SITE_MUST,num_amino_acids,Tier=Tier)
##%% 
#'''
#Create the results of the whole PDBs of Tier_1 and Tier_2
#'''
#from class_results_T_1_and_T_2_PDBs_finalise import results_T_1_and_T_2_PDBs_finalise
#
#saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_PDBs/2020_k_fold_results/",model_name,'/'])
#
#obj_all=results_T_1_and_T_2_PDBs_finalise(T1_T2_test_prob_dic,saving_dir_primary_seq_part)
#obj_all.documentation_prob_all()
#obj_all.documentation_prob_only_given("ONGO",OG_V2_pdbs)
#obj_all.documentation_prob_only_given("TSG",TSG_V2_pdbs)
#obj_all.documentation_prob_only_given("Fusion",Fusion_V2_pdbs)

#%%
Tier =2
names_temp_list=os.listdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/pdb_details_length_source_2020_V_91/T_',str(Tier)]))
saving_dir_primary_seq_part=''.join([ "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_gene/2020_k_fold_results/Tier_",str(Tier),"/",checking_data,model_name,'/'])
for names_temp in names_temp_list:
#names_temp='T_2_Not_Anotated.txt'
    name=names_temp[4:-4]
    if name=='ONGO_TSG' and Tier==1:
        print("Skipped due to overlapping PDB the results of Tier: ",Tier, " of amino_acids: ",num_amino_acids," name ",name)
    else:
        print("creating the results of Tier: ",Tier, " of amino_acids: ",num_amino_acids," name ",name)
        test_gene_direct_prob, test_fin_gene_prob, test_Ensamble_gene_prob = gene_seprated_results_finalise_fully_clean_SITE_MUST_old_threh_T_1_T_2(name,T1_T2_test_prob_dic,saving_dir_primary_seq_part,SITE_MUST,num_amino_acids,Tier=Tier)
