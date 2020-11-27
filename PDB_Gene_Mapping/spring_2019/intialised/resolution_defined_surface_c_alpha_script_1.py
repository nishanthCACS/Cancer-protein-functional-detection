# -*- coding: utf-8 -*-
"""
Created on %25-January-2018(1.08pm)


@author: %A.Nishanth C00294860
loading the experiment type details
"""
import os
import pickle
from resolution_defined_surface_c_alpha_class  import resolution_defined_surface_c_alpha

def pdb_details(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    pdb_train_details = pickle.load(open( ''.join([name,"_training_pdb_ids.p"]), "rb" ))
    pdb_test_details = pickle.load(open( ''.join([name,"_testing_pdb_ids.p"]), "rb" ))
    return pdb_train_details, pdb_test_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
results_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/csv_files_for_NN/"
pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"

#%% For Tier_1
"""
Loading the files with Training and Test SITE information
training and test set creation for 
resolution_xy=3
resolution_xy_z=3
""" 

def results_creation_func(name,loading_dir,results_dir,pdb_source_dir_part):
    resolution_xy = 3
    resolution_xy_z = 3

    results_train_dir= ''.join([results_dir,"3x3_resolution_SITE_must/train/",name])
    results_test_dir=''.join([results_dir,"3x3_resolution_SITE_must/test/",name])
    pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
    pdb_train_details, pdb_test_details = pdb_details(name,loading_dir)
    
    for pdb_train in pdb_train_details:
        for pdb_name in pdb_train:
            obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_train_dir, resolution_xy, resolution_xy_z, pdb_name)
            obj.final_results()
            del obj
    print(" Training results creation done ", name)
    for pdb_test in pdb_test_details:
        for pdb_name in pdb_test:
            obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_test_dir, resolution_xy, resolution_xy_z, pdb_name)
            obj.final_results()
            del obj
    print(" Testing results creation done ", name)

results_creation_func("ONGO",loading_dir,results_dir,pdb_source_dir_part)
results_creation_func("TSG",loading_dir,results_dir,pdb_source_dir_part)
results_creation_func("Fusion",loading_dir,results_dir,pdb_source_dir_part)
#%%

#%% 5x 2 tarining
def results_creation_func(name,loading_dir,results_dir,pdb_source_dir_part):
    resolution_xy = 5
    resolution_xy_z = 2

    results_train_dir= ''.join([results_dir,"5x2_resolution_SITE_must/train/",name])
    results_test_dir=''.join([results_dir,"5x2_resolution_SITE_must/test/",name])
    pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
    pdb_train_details, pdb_test_details = pdb_details(name,loading_dir)
    
    for pdb_train in pdb_train_details:
        for pdb_name in pdb_train:
            obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_train_dir, resolution_xy, resolution_xy_z, pdb_name)
            obj.final_results()
            del obj
    print(" Training results creation done ", name)
    for pdb_test in pdb_test_details:
        for pdb_name in pdb_test:
            obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_test_dir, resolution_xy, resolution_xy_z, pdb_name)
            obj.final_results()
            del obj
    print(" Testing results creation done ", name)

results_creation_func("ONGO",loading_dir,results_dir,pdb_source_dir_part)
results_creation_func("TSG",loading_dir,results_dir,pdb_source_dir_part)
results_creation_func("Fusion",loading_dir,results_dir,pdb_source_dir_part)
#%% 
"""
check the clean UniGene fell in to SITE mut trainning set 
"""
import os
import copy
import shutil
import pickle

clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
loading_dir_train = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
name = "TSG"
os.chdir("/")
os.chdir(loading_dir_train)

train_unigene_details = pickle.load(open( ''.join([name,"_training_Uni_Genes.p"]), "rb" ))
train_pdb_ids_details = pickle.load(open( ''.join([name,"_training_pdb_ids.p"]), "rb" ))
test_unigene_details = pickle.load(open( ''.join([name,"_testing_Uni_Genes.p"]), "rb" ))
test_pdb_ids_details = pickle.load(open( ''.join([name,"_testing_pdb_ids.p"]), "rb" ))
os.chdir("/")
os.chdir(clean_pikle_dir)
SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
clean_unigene_details = pickle.load(open( ''.join([name,"_clean_unigene.p"]), "rb" ) )
Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))

#for i in range(0,len(train_unigene_details)):
#    for j in range(0,len(Thres_sat_gene_list)):
#        if train_unigene_details[i]==Thres_sat_gene_list[j]:
#            if len(train_pdb_ids_details[i])!=len(SITE_unigene_details[j]):
#                print("NOt_Site-train",i,j)
#for i in range(0,len(test_unigene_details)):
#    for j in range(0,len(Thres_sat_gene_list)):
#        if test_unigene_details[i]==Thres_sat_gene_list[j]:
#            if len(test_pdb_ids_details[i])!=len(SITE_unigene_details[j]):
#                print("NOt_Site")
#
#for i in range(0,len(Thres_sat_gene_list)):
#    if Thres_sat_gene_list[i] not in clean_unigene_details:
#        print(Thres_sat_gene_list[i])
#        if  len(SITE_unigene_details) != len(Thres_sat_pdb_list):
#            print("NOT in SITE satisfied ",Thres_sat_gene_list[i])
#%
"""
problemetric files movement while testing
"""
problemetric_unigene = []
problemetric_pdbs =[]
for i in range(0,len(train_unigene_details)):
    if train_unigene_details[i] not in clean_unigene_details:
        print("probelmetric unigene: ",train_unigene_details[i])
        problemetric_unigene.append(copy.deepcopy(train_unigene_details[i]))
        problemetric_pdbs.append(copy.deepcopy(train_pdb_ids_details[i]))
#%
results_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/csv_files_for_NN/"
results_train_dir= ''.join([results_dir,"5x2_resolution_SITE_must/train/",name])
results_test_dir= ''.join([results_dir,"5x2_resolution_SITE_must/test/",name])

dest = ''.join([results_dir,"5x2_problemetric/",name,"/"])
for j in range(0,len(problemetric_pdbs)):
    for i in range(0,len(problemetric_pdbs[j])):
        os.chdir("/")
        os.chdir(results_train_dir)
        #        os.chdir(results_test_dir)
        file_xy = ''.join([problemetric_pdbs[j][i],'_xy.txt'])
        file_yz = ''.join([problemetric_pdbs[j][i],'_yz.txt'])
        file_xz = ''.join([problemetric_pdbs[j][i],'_xz.txt'])
        if os.path.exists(file_xy):
            print("")
            print("Moving: ",problemetric_pdbs[j][i])
            shutil.move(file_xy, dest)          
            shutil.move(file_yz, dest)    
            shutil.move(file_xz, dest)    
#        else:
#            print(problemetric_pdbs[j][i]," already moved")

problemetric_unigene = []
problemetric_pdbs =[]
for i in range(0,len(test_unigene_details)):
    if test_unigene_details[i] not in clean_unigene_details:
        print("probelmetric unigene: ",test_unigene_details[i])
        problemetric_unigene.append(copy.deepcopy(test_unigene_details[i]))
        problemetric_pdbs.append(copy.deepcopy(test_pdb_ids_details[i]))
        
for j in range(0,len(problemetric_pdbs)):
    for i in range(0,len(problemetric_pdbs[j])):
        os.chdir("/")
        os.chdir(results_test_dir)
        file_xy = ''.join([problemetric_pdbs[j][i],'_xy.txt'])
        file_yz = ''.join([problemetric_pdbs[j][i],'_yz.txt'])
        file_xz = ''.join([problemetric_pdbs[j][i],'_xz.txt'])
        if os.path.exists(file_xy): 
            print("")
            print("Moving: ",problemetric_pdbs[j][i])
            shutil.move(file_xy, dest)          
            shutil.move(file_yz, dest)    
            shutil.move(file_xz, dest)    
#        else:
#            print(problemetric_pdbs[j][i]," already moved") 
##%%
#resolution_xy = 3
#resolution_xy_z = 3
#pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
#pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
#for pdb_name in problemetric_pdbs[0]:
#    obj = resolution_defined_surface_c_alpha(pdb_source_dir, dest, resolution_xy, resolution_xy_z, pdb_name)
#    obj.final_results()
#    del obj
#%% count the number of problemtric UniGene and PDBs occurance
import os
import copy
import shutil
import pickle

name = "ONGO"

def stat_of_training_testing(name):
    clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    loading_dir_train = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
    
    os.chdir("/")
    os.chdir(loading_dir_train)
    
    train_unigene_details = pickle.load(open( ''.join([name,"_training_Uni_Genes.p"]), "rb" ))
    train_pdb_ids_details = pickle.load(open( ''.join([name,"_training_pdb_ids.p"]), "rb" ))
    test_unigene_details = pickle.load(open( ''.join([name,"_testing_Uni_Genes.p"]), "rb" ))
    test_pdb_ids_details = pickle.load(open( ''.join([name,"_testing_pdb_ids.p"]), "rb" ))
    os.chdir("/")
    os.chdir(clean_pikle_dir)
    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    clean_unigene_details = pickle.load(open( ''.join([name,"_clean_unigene.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    problemetric_unigene = []
    problemetric_pdbs =[]
    
    print("")
    print("------- considering:  ",name, " -----------------------------")
    print("")
    print("Number of Gene's took for training:",len(train_unigene_details))
    print("Number of Gene's took for testing:",len(test_unigene_details))
    
    for i in range(0,len(train_unigene_details)):
        if train_unigene_details[i] not in clean_unigene_details:
            print("probelmetric unigene train: ",train_unigene_details[i])
            problemetric_unigene.append(copy.deepcopy(train_unigene_details[i]))
            problemetric_pdbs.append(copy.deepcopy(train_pdb_ids_details[i]))
            print("probelmetric pdbs length train: ",len(train_pdb_ids_details[i]))
            
    problemetric_unigene = []
    problemetric_pdbs =[]
    for i in range(0,len(test_unigene_details)):
        if test_unigene_details[i] not in clean_unigene_details:
            print("probelmetric unigene test: ",test_unigene_details[i])
            problemetric_unigene.append(copy.deepcopy(test_unigene_details[i]))
            problemetric_pdbs.append(copy.deepcopy(test_pdb_ids_details[i]))
            print("probelmetric pdbs length test: ",len(test_pdb_ids_details[i]))
    
stat_of_training_testing("ONGO")
stat_of_training_testing("TSG")
stat_of_training_testing("Fusion")
#%%#%% selected atoms
"""
Finally random selected atoms for train and test
"""
name ="ONGO"

ONGO_unigenes_for_test = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']

unigenes_for_test = ONGO_unigenes_for_test
#TSG_unigenes_for_test = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
#Fusion_unigenes_for_test = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
loading_dir_train = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"


os.chdir("/")
os.chdir(clean_pikle_dir)
SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
clean_unigene_details = pickle.load(open( ''.join([name,"_clean_unigene.p"]), "rb" ) )
Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))

for i in range(0,len(unigenes_for_test)):
    if unigenes_for_test[i] not in clean_unigene_details:
        print("Not in clean ",u_g)
        del unigenes_for_test[i]
    
# for training UniGene


