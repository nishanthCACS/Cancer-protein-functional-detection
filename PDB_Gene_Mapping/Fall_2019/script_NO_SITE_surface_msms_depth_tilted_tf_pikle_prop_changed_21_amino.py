# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import pickle
import os
#%%
def create_train_PDBs_list(name):
#    name='ONGO'    
    
    if name=='TSG':
        loading_thresh_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
        os.chdir("/")
        os.chdir(loading_thresh_dir)
#        pdbs_not_software_must = pickle.load(open(''.join([name,"_pdbs_not_software_must.p"]), "rb" ))
        pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_must_gene.p"]), "rb")) 

    saving_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/results/"   
    
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
        '''No Hs.517792'''
        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
        if pdbs_not_software_must_gene[0] in gene_test_details:
            raise("chcek the TSG")
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

#%%
name = "ONGO"
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = create_train_PDBs_list(name)
train_pdb_ids_details_all=sum(train_pdb_ids_details,[])

#%%
name = "TSG"
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = create_train_PDBs_list(name)
train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
#%%

name = "Fusion"
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = create_train_PDBs_list(name)
train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
#%%
from  class_surface_msms_depth_tilted_tf_pikle_prop_changed_21_amino import surface_msms_depth_class

def pdb_details_fun(name):
#    name='ONGO'    
    loading_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/results/"   

    os.chdir("/")
    os.chdir(loading_dir)
    test_pdb_ids_details  = pickle.load(open( ''.join([name,"_test_pdb_ids_details.p"]), "rb" ) )
    train_pdb_ids_details =  pickle.load(open( ''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
    test_unigene_details =  pickle.load(open( ''.join([name,"_test_unigene_details.p"]), "rb" )) 
    return train_pdb_ids_details,test_pdb_ids_details,test_unigene_details
    
def creation_of_data(name,height):
#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    
    train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/unigene/optimaly_tilted"   
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/neg_size_19_21_aminoacids"
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/neg_size_20_21_aminoacids"
#    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/neg_size_15_21_aminoacids"
    saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/unigene_Optimally_tilted/21_aminoacids"

    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
    saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])
    train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
    print(name," training in progress")
    
#    pdb_chk_list=['1W72','2AK4','2H26','4PJE','5C08','5C0A','5C0B','5C0C','5EUO','5U6Q']
#    for pdb_name in pdb_chk_list:
#        print(pdb_name)
#        obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_train, pdb_name,height=height)
#        del obj
    
    for pdb_name in train_pdb_ids_details_all:
        if pdb_name!='4MDQ' and pdb_name!='6AXG':
#            print(pdb_name)
            obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_train, pdb_name,height=height)
            del obj
    
    saving_dir_part_test=''.join([saving_dir_part,'/Test/',name])
    test_pdb_ids_details_all=sum(test_pdb_ids_details,[])
    print(name," testing in progress")
    
    for pdb_name in test_pdb_ids_details_all:
        if pdb_name!='4MDQ' and pdb_name!='6AXG':
#            print(pdb_name)
            obj = surface_msms_depth_class(loading_pikle_dir, saving_dir_part_test, pdb_name,height=height)
            del obj    
#%%
height=200
name = "ONGO"
creation_of_data(name,height)
#%
#name = "TSG"
#creation_of_data(name,height)
##
name = "Fusion"
creation_of_data(name,height)

#%%
for i in range(0,len(train_pdb_ids_details)):
    for j in range(0,len(train_pdb_ids_details[i])):
        if train_pdb_ids_details[i][j]=='6AXG':
            print(i)
#%%
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)