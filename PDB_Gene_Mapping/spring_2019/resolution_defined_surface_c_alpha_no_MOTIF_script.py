# -*- coding: utf-8 -*-
"""
Created on %25-January-2018(1.08pm)


@author: %A.Nishanth C00294860
loading the experiment type details
"""
import os
import pickle
from resolution_defined_surface_c_alpha_no_MOTIF_class  import resolution_defined_surface_c_alpha
#%%
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
results_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/csv_files_for_NN/"
pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
"""
    create the dataset for ony considering TSG, ONGO conflict
"""
def pdb_details_conf_ONGO_TSG(name,clean_pikle_dir):
    os.chdir("/")
    os.chdir(clean_pikle_dir)
#    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    if name=="ONGO":
        gene_test_details = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="TSG":
        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="Fusion":
        gene_test_details = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
        clean_unigene_details = Thres_sat_gene_list
    #% first select the training UniGenes
    gene_train_details = []
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] not in gene_test_details:
            if Thres_sat_gene_list[i] in clean_unigene_details:
                gene_train_details.append(Thres_sat_gene_list[i])

    #% check the test list is any UniGene fell into the conflict
    for u_g in gene_test_details:
        if u_g in clean_unigene_details:
            print("cut the UniGene ",u_g," from the test_Uni_genes")
    pdb_train_details=[]
    pdb_test_details =[]
    for i in range(0,len(Thres_sat_gene_list)):
         if Thres_sat_gene_list[i] in gene_train_details:
             pdb_train_details.append(Thres_sat_pdb_list[i])
    for i in range(0,len(Thres_sat_gene_list)):
         if Thres_sat_gene_list[i] in gene_test_details:
             pdb_test_details.append(Thres_sat_pdb_list[i])
              
    return  pdb_train_details, pdb_test_details     

clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
loading_dir_train = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"

clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
loading_dir_train = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"

def results_creation_func(name,loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir):
#    resolution_xy = 3
#    resolution_xy_z = 3
    resolution_xy = 3
    resolution_xy_z = 1
    results_train_dir= ''.join([results_dir,"3x1/train/",name])
    results_test_dir=''.join([results_dir,"3x1/test/",name])
    results_prob_dir= ''.join([results_dir,"3x1_prob/",name])
    pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
    pdb_train_details, pdb_test_details = pdb_details_conf_ONGO_TSG(name,clean_pikle_dir)
    
    os.chdir("/")
    os.chdir(clean_pikle_dir)
    if name == "Fusion":
        clean_unigene_details = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    else:
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_PDBs.p"]), "rb" ) )
    clean_unigene_details =sum(clean_unigene_details, [])
    for pdb_train in pdb_train_details:
        for pdb_name in pdb_train:
            if pdb_name in clean_unigene_details:
                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_train_dir, resolution_xy, resolution_xy_z, pdb_name)
                obj.final_results()
                del obj
            else:
                print("Problemetric PDB in train: ",pdb_name)
                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
                obj.final_results()
                del obj
    print(" Training results creation done ", name)
    for pdb_test in pdb_test_details:
        for pdb_name in pdb_test:
            if pdb_name in clean_unigene_details:
                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_test_dir, resolution_xy, resolution_xy_z, pdb_name)
                obj.final_results()
                del obj
            else:
                print("Problemetric PDB in test: ",pdb_name)
                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
                obj.final_results()
                del obj
    


results_creation_func("ONGO",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
results_creation_func("TSG",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
results_creation_func("Fusion",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)

#%% saving training tyesting details

def saving_training_test_pikles(name):
    clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    os.chdir("/")
    os.chdir(clean_pikle_dir)
#    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    if name=="ONGO":
        gene_test_details = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="TSG":
        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="Fusion":
        gene_test_details = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
        clean_unigene_details = Thres_sat_gene_list
    #% first select the training UniGenes
    gene_train_details = []
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] not in gene_test_details:
            if Thres_sat_gene_list[i] in clean_unigene_details:
                    gene_train_details.append(Thres_sat_gene_list[i])
            else:
                print("Missed_gene: ",Thres_sat_gene_list[i])
                print("length_threshold: ",len(Thres_sat_pdb_list[i]))
    unigene_test_details=[]
    pdb_test_details =[]
    for i in range(0,len(Thres_sat_gene_list)):
         if Thres_sat_gene_list[i] in gene_test_details:
             pdb_test_details.append(Thres_sat_pdb_list[i])
             unigene_test_details.append(Thres_sat_gene_list[i])
            
    unigene_train_details=[]
    pdb_train_details =[]
    for i in range(0,len(Thres_sat_gene_list)):
         if Thres_sat_gene_list[i] in gene_train_details:
             pdb_train_details.append(Thres_sat_pdb_list[i])
             unigene_train_details.append(Thres_sat_gene_list[i])

   
    saving_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean"
    os.chdir("/")
    os.chdir(saving_pikle_dir)
    pickle.dump(unigene_test_details, open(''.join([name,"_test_uni_gene.p"]), "wb" ))
    pickle.dump(pdb_test_details, open(''.join([name,"_test_pdb_ids.p"]), "wb" ))
    pickle.dump(unigene_train_details, open(''.join([name,"_train_uni_gene.p"]), "wb" ))
    pickle.dump(pdb_train_details, open(''.join([name,"_train_pdb_ids.p"]), "wb" ))
saving_training_test_pikles("ONGO")          
print("TSG")
saving_training_test_pikles("TSG") 
print("Fusion")
saving_training_test_pikles("Fusion")                        

#%%
#def pdb_details(name,loading_dir):
#    os.chdir("/")
#    os.chdir(loading_dir)
#    pdb_train_details = pickle.load(open( ''.join([name,"_training_pdb_ids.p"]), "rb" ))
#    pdb_test_details = pickle.load(open( ''.join([name,"_testing_pdb_ids.p"]), "rb" ))
#    return pdb_train_details, pdb_test_details
#
#loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
#results_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/csv_files_for_NN/"
#pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
#clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
##%% For Tier_1
#"""
#Loading the files with Training and Test SITE information
#training and test set creation for 
#resolution_xy=3
#resolution_xy_z=3
#""" 
#def results_creation_func(name,loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir):
#    resolution_xy = 3
#    resolution_xy_z = 3
#    
#    os.chdir("/")
#    os.chdir(clean_pikle_dir)
#    clean_unigene_details = (sum(pickle.load(open( ''.join([name,"_clean_PDBs.p"]), "rb" ) ), []))
#
#    results_train_dir= ''.join([results_dir,"3x3_resolution_SITE_must/train/",name])
#    results_test_dir=''.join([results_dir,"3x3_resolution_SITE_must/test/",name])
#    results_prob_dir= ''.join([results_dir,"3x3_problemetric"])
#    pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
#    pdb_train_details, pdb_test_details = pdb_details(name,loading_dir)
#    
#    for pdb_train in pdb_train_details:
#        for pdb_name in pdb_train:
#            if pdb_name in clean_unigene_details:
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_train_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#            else:
#                print("Problemetric PDB in train: ",pdb_name)
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#    print(" Training results creation done ", name)
#    for pdb_test in pdb_test_details:
#        for pdb_name in pdb_test:
#            if pdb_name in clean_unigene_details:
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_test_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#            else:
#                print("Problemetric PDB in test: ",pdb_name)
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#    print(" Testing results creation done ", name)
#
#results_creation_func("ONGO",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#results_creation_func("TSG",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#results_creation_func("Fusion",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#
##%% selected atoms
#"""
#Finally random selected atoms for train and test
#"""
#def pdb_details(name,clean_pikle_dir):
#    os.chdir("/")
#    os.chdir(clean_pikle_dir)
##    clean_unigene_details = pickle.load(open( ''.join([name,"_clean_unigene.p"]), "rb" ) )
#    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
#    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
#    if name=="ONGO":
#        gene_test_details = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
#    elif name =="TSG":
#        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
#    elif name =="Fusion":
#        gene_test_details = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
#
#    #% first select the training UniGenes
#    gene_train_details = []
#    for i in range(0,len(Thres_sat_gene_list)):
#        if Thres_sat_gene_list[i] not in gene_test_details:
#            gene_train_details.append(Thres_sat_gene_list[i])
#                
#    pdb_train_details=[]
#    pdb_test_details =[]
#    for i in range(0,len(Thres_sat_gene_list)):
#         if Thres_sat_gene_list[i] in gene_train_details:
#             pdb_train_details.append(Thres_sat_pdb_list[i])
#    for i in range(0,len(Thres_sat_gene_list)):
#         if Thres_sat_gene_list[i] in gene_test_details:
#             pdb_test_details.append(Thres_sat_pdb_list[i])
#                    
#    return  pdb_train_details, pdb_test_details            
#
#clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
#loading_dir_train = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
#
#def results_creation_func(name,loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir):
#    resolution_xy = 3
#    resolution_xy_z = 3
#    
#    os.chdir("/")
#    os.chdir(clean_pikle_dir)
#    clean_unigene_details = (sum(pickle.load(open( ''.join([name,"_clean_PDBs.p"]), "rb" ) ), []))
#
#    results_train_dir= ''.join([results_dir,"3x3/train/",name])
#    results_test_dir=''.join([results_dir,"3x3/test/",name])
#    results_prob_dir= ''.join([results_dir,"3x3_prob/",name])
#    pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
#    pdb_train_details, pdb_test_details = pdb_details(name,clean_pikle_dir)
#    
#    for pdb_train in pdb_train_details:
#        for pdb_name in pdb_train:
#            if pdb_name in clean_unigene_details:
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_train_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#            else:
#                print("Problemetric PDB in train: ",pdb_name)
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#    print(" Training results creation done ", name)
#    for pdb_test in pdb_test_details:
#        for pdb_name in pdb_test:
#            if pdb_name in clean_unigene_details:
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_test_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#            else:
#                print("Problemetric PDB in test: ",pdb_name)
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#    print(" Testing results creation done ", name)
#
#results_creation_func("ONGO",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#results_creation_func("TSG",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#results_creation_func("Fusion",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#
##% 5x 2 tarining
#def results_creation_func_2(name,loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir):
#    resolution_xy = 5
#    resolution_xy_z = 2
#    
#    os.chdir("/")
#    os.chdir(clean_pikle_dir)
#    clean_unigene_details = (sum(pickle.load(open( ''.join([name,"_clean_PDBs.p"]), "rb" ) ), []))
#
#    results_train_dir= ''.join([results_dir,"5x2/train/",name])
#    results_test_dir=''.join([results_dir,"5x2/test/",name])
#    results_prob_dir= ''.join([results_dir,"5x2_prob/",name])
#    
#    pdb_source_dir = ''.join([pdb_source_dir_part, name, "_R"])
#    pdb_train_details, pdb_test_details = pdb_details(name,clean_pikle_dir)
#    
#    for pdb_train in pdb_train_details:
#        for pdb_name in pdb_train:
#            if pdb_name in clean_unigene_details:
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_train_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#            else:
#                print("Problemetric PDB in train: ",pdb_name)
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#    print(" Training results creation done ", name)
#    for pdb_test in pdb_test_details:
#        for pdb_name in pdb_test:
#            if pdb_name in clean_unigene_details:
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_test_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#            else:
#                print("Problemetric PDB in test: ",pdb_name)
#                obj = resolution_defined_surface_c_alpha(pdb_source_dir, results_prob_dir, resolution_xy, resolution_xy_z, pdb_name)
#                obj.final_results()
#                del obj
#    print(" Testing results creation done ", name)
#
#results_creation_func_2("ONGO",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#results_creation_func_2("TSG",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)
#results_creation_func_2("Fusion",loading_dir,results_dir,pdb_source_dir_part,clean_pikle_dir)