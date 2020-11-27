# -*- coding: utf-8 -*-
"""
Created on %23-Feb-2019(12.59P.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle
from random import shuffle

from class_surface_msms_depth  import surface_msms_depth_class
#%%
def pdb_details_fun(name):
#    name='ONGO'    
    saving_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/all_results/unigene"   
    
    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    
    os.chdir("/")
    os.chdir(site_dir)
#    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    
    clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    os.chdir("/")
    os.chdir(clean_pikle_dir)
    if name=="ONGO":
#        gene_test_details = ['Hs.731652','Hs.37003','Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
        gene_test_details = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="TSG":
        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="Fusion":
        gene_test_details = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
        clean_unigene_details = Thres_sat_gene_list
        
    train_unigene_details=[]       
    test_unigene_details=[]
    train_pdb_ids_details=[]
    test_pdb_ids_details=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] in clean_unigene_details:
            if Thres_sat_gene_list[i] not in gene_test_details:
                train_pdb_ids_details.append(Thres_sat_pdb_list[i])
                train_unigene_details.append(Thres_sat_gene_list[i])
            else:
                test_pdb_ids_details.append(Thres_sat_pdb_list[i])
                test_unigene_details.append(Thres_sat_gene_list[i])
    os.chdir('/')
    os.chdir(saving_dir)    
    pickle.dump(train_pdb_ids_details, open(''.join([name,"_train_pdb_ids_details.p"]), "wb" ))
    pickle.dump(test_pdb_ids_details, open(''.join([name,"_test_pdb_ids_details.p"]), "wb" ))
    pickle.dump(test_unigene_details, open(''.join([name,"_test_unigene_details.p"]), "wb" ))  
    return train_pdb_ids_details,test_pdb_ids_details,test_unigene_details

def creation_of_data(name):
#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    
    train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/all_results/unigene"   
    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
    saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])
    train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
    print(name," training in progress")
    for pdb_name in train_pdb_ids_details_all:
        if pdb_name!='4MDQ' and  pdb_name!='6AXG':
            obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_train, pdb_name)
            del obj
    
    saving_dir_part_test=''.join([saving_dir_part,'/Test/',name])
    test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
    print(name," testing in progress")
    
    for pdb_name in test_pdb_ids_details_all:
        if pdb_name!='4MDQ' and  pdb_name!='6AXG':
            obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_test, pdb_name)
            del obj    
#%%
name = "ONGO"
creation_of_data(name)
name = "TSG"
creation_of_data(name)
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
#saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/unigene"   
#
#train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
#test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
#
name = "Fusion"
creation_of_data(name)
#train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
#saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/unigene"   
#
#train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
#test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
#%%
'''create the dataset with mixed pdbs'''
def pdb_details_fun_mixed(name):
    saving_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/all_results/mixed_pdbs"   
    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    
    os.chdir("/")
    os.chdir(site_dir)
#    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    
    clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    os.chdir("/")
    os.chdir(clean_pikle_dir)
    if name=="ONGO":
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="TSG":
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="Fusion":
        clean_unigene_details = Thres_sat_gene_list
        
    pdb_list_all_sat=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] in clean_unigene_details:
            pdb_list_all_sat.append(Thres_sat_pdb_list[i])
                
    pdb_list_all_sat_all = sum(pdb_list_all_sat,[])
    # then choose the training data
    if name=='Fusion':
        number_training_pdbs=365
    else:
        number_training_pdbs=1000
        
    shuffle(pdb_list_all_sat_all)
    train_pdb_ids_details = []
    for i in range(0,number_training_pdbs):
        train_pdb_ids_details.append(pdb_list_all_sat_all[i])
        
    test_pdb_ids_details=[]
    for i in range(number_training_pdbs,len(pdb_list_all_sat_all)):
        test_pdb_ids_details.append(pdb_list_all_sat_all[i])

    os.chdir(saving_dir)    
    pickle.dump(train_pdb_ids_details, open(''.join([name,"_train_pdb_ids_details.p"]), "wb" ))
    pickle.dump(test_pdb_ids_details, open(''.join([name,"_test_pdb_ids_details.p"]), "wb" ))
    
    return train_pdb_ids_details,test_pdb_ids_details

def creation_of_data_mixed(name):
#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    
    train_pdb_ids_details,test_pdb_ids_details = pdb_details_fun_mixed(name)
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/all_results/mixed_pdbs"   
    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
    saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])

    print(name," training in progress")
    for pdb_name in train_pdb_ids_details:
        if pdb_name!='4MDQ' and  pdb_name!='6AXG':
            obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_train, pdb_name)
            del obj
    
    saving_dir_part_test=''.join([saving_dir_part,'/Test/',name])
    print(name," testing in progress")
    
    for pdb_name in test_pdb_ids_details:
        if pdb_name!='4MDQ' and  pdb_name!='6AXG':
            obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_test, pdb_name)
            del obj    
        
creation_of_data_mixed('ONGO')
creation_of_data_mixed('TSG')
creation_of_data_mixed('Fusion')
#%% problem with the threshold condition
name="ONGO"
pdb_name='721P'
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
os.chdir('/')
os.chdir(loading_pikle_dir)
coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))         