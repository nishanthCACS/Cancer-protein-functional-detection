# -*- coding: utf-8 -*-
"""
Created on %07-April-2019(11.59A.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle

from class_surface_msms_depth  import surface_msms_depth_MOTIF_class

#%%
'''create the dataset with test pdbs of different classes'''
def pdb_details_fun_finalise(name):
    '''
    This function retrieve the SITE_information and gene informaation and select the PDBs(with the gene information) to test
    And the _SITE_missed_genes: contains the inforfation about the genes that doesn't have single PDB has SITE information
    But these genes atleast has one PDB which satisfiedd threshold condition
    '''
    saving_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/after_success"   
    site_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked"
    
    os.chdir("/")
    os.chdir(site_dir)
    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    
    pdb_list_all_sat=[]
    test_uni_gene=[]
    missed_genes=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if len(SITE_unigene_details[i])>0:
            pdb_list_all_sat.append(SITE_unigene_details[i])
            test_uni_gene.append(Thres_sat_gene_list[i])
        else:
            missed_genes.append(Thres_sat_gene_list[i])
            
    test_pdb_ids_details = sum(pdb_list_all_sat,[])

    os.chdir('/')
    os.chdir(saving_dir)    
    pickle.dump(test_uni_gene, open(''.join([name,"_test_uni_gene.p"]), "wb" ))
    pickle.dump(test_pdb_ids_details, open(''.join([name,"_test_pdb_ids_details.p"]), "wb" ))
    pickle.dump(missed_genes, open(''.join([name,"_SITE_missed_genes.p"]), "wb" ))

    return test_uni_gene,test_pdb_ids_details

def creation_of_data_final(name):
    """
    This creates the source files for NN
    """
    #here all the files NN_source files are saved under same directory 
    saving_dir_part_test = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/after_success/NN_source"
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/after_success/"
    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
    test_uni_gene,test_pdb_ids_details = pdb_details_fun_finalise(name)
    
    for pdb_name in test_pdb_ids_details:
        obj = surface_msms_depth_MOTIF_class(loading_pikle_dir, saving_dir_part_test, pdb_name)
        del obj  

            
#%%
working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
os.chdir('/')
os.chdir(working_dir)
files=os.listdir()
#%%
for names in files:
    creation_of_data_final(names[8:-4])
    print("creation_of_data_final ",names[8:-4] ,"done :)")

#%% For varification purpose
'''
def used_detail_verify(name,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details):
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean')
    test_gene_list_used = pickle.load(open(''.join([name,'_test_uni_gene.p']),"rb"))
    test_pdb_list_used = pickle.load(open(''.join([name,'_test_pdb_ids_details.p']),"rb"))
    train_pdb_list_used = pickle.load(open(''.join([name,'_train_pdb_ids_details.p']),"rb"))
    if test_gene_list_used==test_unigene_details:
        print('1')
        if test_pdb_list_used==test_pdb_ids_details:
            print('2')
            if train_pdb_ids_details==train_pdb_list_used:
                print(name,' done correctly')
#    else:
#        return test_gene_list_used,test_pdb_list_used,train_pdb_list_used
#%%
name = "ONGO"
train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
used_detail_verify(name,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details)

name = "TSG"
train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
used_detail_verify(name,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details)
name = "Fusion"
train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
used_detail_verify(name,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details)
'''