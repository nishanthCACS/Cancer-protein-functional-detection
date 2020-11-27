# -*- coding: utf-8 -*-
"""
Created on %15-November-2018(16.09pm)

@author: %A.Nishanth C00294860

cleaning the overlapping PDBs
"""
import pickle
import os
import copy
#UniGenes_for_train = ['Hs.719495', 'Hs.479756','Hs.706627','Hs.726012']
#UniGenes_for_test = ['Hs.97439','Hs.506852']

traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
name = "ONGO"
def retrieve_ids(name,traing_testing_pdb_dir):

    ongo_uni_gene_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
    os.chdir("/")
    os.chdir(ongo_uni_gene_pdb_dir)
    unigene = pickle.load(open(''.join([name,"_thres_sat_gene_list.p"]), "rb" ) )
    unigene_PDB = pickle.load(open(''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ) )
    return unigene,unigene_PDB

OG_unigene,OG_unigene_PDB =  retrieve_ids("ONGO",traing_testing_pdb_dir)
TSG_unigene,TSG_unigene_PDB =  retrieve_ids("TSG",traing_testing_pdb_dir)
Fusion_unigene,Fusion_unigene_PDB =  retrieve_ids("Fusion",traing_testing_pdb_dir)
#%% then check the overlapping between them
OG_pdbs = sum(OG_unigene_PDB, [])
TSG_pdbs = sum(TSG_unigene_PDB, []) 
Fusion_pdbs = sum(Fusion_unigene_PDB, []) 

intersection_PDBs = list(set(OG_pdbs) & set(TSG_pdbs))
#% overlap between Fusion and OG
intersection_PDBs_OF = list(set(OG_pdbs) & set(Fusion_pdbs))
#% overlap between Fusion and TSG
intersection_PDBs_TF = list(set(TSG_pdbs) & set(Fusion_pdbs))
#% then select the PDBs and remove the whole overlapping PDBs
def removing_list(intersection_PDBs,intersection_PDBs_set):
    """
    Merge the two lists to form a new list for removing
    """
    removed = copy.deepcopy(intersection_PDBs)
    for i in intersection_PDBs_set:
        removed.append(i)
    return removed

og_unwanted_PDBs = removing_list(intersection_PDBs,intersection_PDBs_OF)
tsg_unwanted_PDBs = removing_list(intersection_PDBs,intersection_PDBs_TF)
fus_unwanted_PDBs = removing_list(intersection_PDBs_OF,intersection_PDBs_TF)
#%% then go through the PDBs and remove the overlapping PDBs
def cleaned_PDB_ids(name, unigene_PDB, unwanted_PDBs):
    cleaned_unigene_PDB = []   
    for PDB_list in unigene_PDB: 
        PDB_temp = []
        for i in PDB_list:
            if i not in unwanted_PDBs:
                PDB_temp.append(i)
        cleaned_unigene_PDB.append(PDB_temp)
    return cleaned_unigene_PDB

OG_clean_unigene_PDB = cleaned_PDB_ids("ONGO", OG_unigene_PDB, og_unwanted_PDBs)
TSG_clean_unigene_PDB = cleaned_PDB_ids("TSG", TSG_unigene_PDB, tsg_unwanted_PDBs)
Fusion_clean_unigene_PDB = cleaned_PDB_ids("Fusion", Fusion_unigene_PDB, fus_unwanted_PDBs)
#%% if the UniGene doesn't have PDBs just remove them
def create_clean_gene_pdbs(name,unigene,clean_unigene_PDB,unwanted_PDBs):
    clean_UniGene = []
    clean_only_unigene_PDBs = []
    for i in range(0,len(clean_unigene_PDB)):
        if len(clean_unigene_PDB[i])>0:
            clean_UniGene.append(unigene[i])
            clean_only_unigene_PDBs.append(clean_unigene_PDB[i])
            
    """
    These pikles canbe used to avoid the classifiction issues later created by the overlapping PDBs
    thus ONE can avoid these PDBs toi classify or use together
    """
    pickle.dump(clean_only_unigene_PDBs, open(''.join([name,"_clean_unigene_PDB.p"]), "wb" ) ) 
    pickle.dump(unwanted_PDBs, open(''.join([name,"_overlapped_PDBs.p"]), "wb" ) )
    pickle.dump(clean_UniGene, open(''.join([name,"_clean_UniGene.p"]), "wb" ) )
    return clean_UniGene, clean_only_unigene_PDBs

uni_gene_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
os.chdir("/")
os.chdir(uni_gene_pdb_dir)

OG_clean_UniGene, OG_clean_only_unigene_PDBs = create_clean_gene_pdbs("ONGO",OG_unigene,OG_clean_unigene_PDB,og_unwanted_PDBs)
TSG_clean_UniGene, TSG_clean_only_unigene_PDBs = create_clean_gene_pdbs("TSG",TSG_unigene,TSG_clean_unigene_PDB,tsg_unwanted_PDBs)
Fusion_clean_UniGene, Fusion_clean_only_unigene_PDBs = create_clean_gene_pdbs("Fusion",Fusion_unigene,Fusion_clean_unigene_PDB,fus_unwanted_PDBs)
#%%
"""

Data creation starts here!!!!!!!!!!!!!!!!!!

"""
"""
first load the PDB_ids used in new_ran_dataset for training and testing and remove the overlapping(unwanted) PDB_ids     
"""
def get_PDB_ids(name,unigene,unigene_PDB,traing_testing_pdb_dir):
    """
    Here
    
    unigene: cleaned unigene ids
    unigene_PDB: cleaned unigene to PDB_ids
    """
    os.chdir("/")
    os.chdir(traing_testing_pdb_dir)
    whole_training_unigene = pickle.load(open(''.join([name,"_train_uni_gene.p"]), "rb" ) ) 
    # testing data pickle
    whole_testing_unigene = pickle.load(open(''.join([name,"_test_uni_gene.p"]), "rb" ) ) 

    whole_training_pdb_ids = []
    for u in whole_training_unigene:
        for i in range(0,len(unigene)):
            if unigene[i] == u:
                whole_training_pdb_ids.append(unigene_PDB[i])
    whole_testing_pdb_ids= []
    for u in whole_testing_unigene:
        for i in range(0,len(unigene)):
            if unigene[i] == u:
                whole_testing_pdb_ids.append(unigene_PDB[i])    
    return whole_training_pdb_ids, whole_testing_pdb_ids,whole_training_unigene,whole_testing_unigene
#traing_testing_pdb_dir_tf = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/new_random_whole"    
traing_testing_pdb_dir_tf = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/new_random_whole"    

TSG_whole_training_pdb_ids, TSG_whole_testing_pdb_ids,TSG_whole_training_unigene, TSG_whole_testing_unigene = get_PDB_ids("TSG",TSG_clean_UniGene,TSG_clean_only_unigene_PDBs,traing_testing_pdb_dir_tf)
Fusion_whole_training_pdb_ids, Fusion_whole_testing_pdb_ids, Fusion_whole_training_unigene, Fusion_whole_testing_unigene = get_PDB_ids("Fusion",Fusion_clean_UniGene,Fusion_clean_only_unigene_PDBs,traing_testing_pdb_dir_tf)

#%% then check the already created datasets to remove the unwanted PDBs
import shutil

def create_data_set(name,whole_training_pdb_ids,whole_testing_pdb_ids,whole_training_unigene,whole_testing_unigene):
    """
    Inorder to copy the files to copy the results
    
    """
    
    saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
    saving_dir_train =  ''.join([saving_pik, "/train/",name])
    saving_dir_test =  ''.join([saving_pik, "/test/",name])
    # check whether the apth exists or not and create the directory if not there
    if not os.path.isdir(saving_dir_train):
        os.makedirs(saving_dir_train)      
        print(saving_dir_train, " path created")
    if not os.path.isdir(saving_dir_test):
        os.makedirs(saving_dir_test)      
        print(saving_dir_test, " path created")
            
    train_pdbs = sum(whole_training_pdb_ids, [])
    test_pdbs = sum(whole_testing_pdb_ids, [])
    
    # since these surface results is already calculated so copy from that
    source_results_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/",name])
    os.chdir("/")
    os.chdir(source_results_dir)
    missed_pdbs_train =[]
    for pdb_name in train_pdbs:
        fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
        if os.path.isfile(fname): 
            file_xy = ''.join([pdb_name,'_xy.txt'])
            file_yz = ''.join([pdb_name,'_yz.txt'])
            file_xz = ''.join([pdb_name,'_xz.txt'])
            shutil.copy(file_xy, saving_dir_train)          
            shutil.copy(file_yz, saving_dir_train)    
            shutil.copy(file_xz, saving_dir_train)
        else:
            missed_pdbs_train.append(pdb_name)
    missed_pdbs_test = []
    for pdb_name in test_pdbs:
        fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
        if os.path.isfile(fname): 
            file_xy = ''.join([pdb_name,'_xy.txt'])
            file_yz = ''.join([pdb_name,'_yz.txt'])
            file_xz = ''.join([pdb_name,'_xz.txt'])
            shutil.copy(file_xy, saving_dir_test)          
            shutil.copy(file_yz, saving_dir_test)    
            shutil.copy(file_xz, saving_dir_test)
        else:
            missed_pdbs_test.append(pdb_name)
    #%save the created randomised training unigene pdb ids as pikle files
    os.chdir("/")
    os.chdir(saving_pik)
    pickle.dump(whole_training_unigene, open( ''.join([name,"_train_uni_gene.p"]), "wb" ))
    pickle.dump(whole_training_pdb_ids, open( ''.join([name,"_train_pdb_ids.p"]), "wb" ) )  
    pickle.dump(whole_testing_unigene, open( ''.join([name,"_test_unoverlap_uni_gene.p"]), "wb" ))
    pickle.dump(whole_testing_pdb_ids, open( ''.join([name,"_test_unoverlap_pdb_ids.p"]), "wb" ) )  
    print(name, " created :)")

#%% creating dataset with unoverlapping PDBs
create_data_set("TSG",TSG_whole_training_pdb_ids, TSG_whole_testing_pdb_ids,TSG_whole_training_unigene, TSG_whole_testing_unigene)
create_data_set("Fusion",Fusion_whole_training_pdb_ids, Fusion_whole_testing_pdb_ids, Fusion_whole_training_unigene, Fusion_whole_testing_unigene)

#%% 
"""then create the Dataset for ONGO"""
  
traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
name = "ONGO"
#%% then create the whole set
OG_whole_testing_unigenes = ["Hs.162233","Hs.365116","Hs.707742","Hs.165950","Hs.338207","Hs.470316","Hs.486502","Hs.507590","Hs.515247","Hs.525622","Hs.591742","Hs.631535","Hs.715871"]
for i in OG_whole_testing_unigenes:
    if i not in OG_clean_UniGene:
        print(i ," not in OG_clean_UniGene")

OG_whole_testing_pdb_ids = []
for i in OG_whole_testing_unigenes:
    for u in range(0,len(OG_clean_UniGene)):
        if OG_clean_UniGene[u] == i:
            OG_whole_testing_pdb_ids.append(OG_clean_only_unigene_PDBs[u])
#% then choose the rest of the UniGenes as training
OG_whole_training_unigenes = []
for i in OG_clean_UniGene:
    if i not in OG_whole_testing_unigenes:         
        OG_whole_training_unigenes.append(i)

OG_whole_training_pdb_ids = []
for i in OG_whole_training_unigenes:
    for u in range(0,len(OG_clean_UniGene)):
        if OG_clean_UniGene[u] == i:
            OG_whole_training_pdb_ids.append(OG_clean_only_unigene_PDBs[u])
#%          
saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/cleaned"
saving_dir_train =  ''.join([saving_pik, "/train/",name])
saving_dir_test =  ''.join([saving_pik, "/test/",name])
# check whether the apth exists or not and create the directory if not there
if not os.path.isdir(saving_dir_train):
    os.makedirs(saving_dir_train)      
    print(saving_dir_train, " path created")
if not os.path.isdir(saving_dir_test):
    os.makedirs(saving_dir_test)      
    print(saving_dir_test, " path created")
        
train_pdbs = sum(OG_whole_training_pdb_ids, [])
test_pdbs = sum(OG_whole_testing_pdb_ids, [])

# since these surface results is already calculated so copy from that
source_results_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/whole/",name])
os.chdir("/")
os.chdir(source_results_dir)
missed_pdbs_train =[]
for pdb_name in train_pdbs:
    fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
    if os.path.isfile(fname): 
        file_xy = ''.join([pdb_name,'_xy.txt'])
        file_yz = ''.join([pdb_name,'_yz.txt'])
        file_xz = ''.join([pdb_name,'_xz.txt'])
        shutil.copy(file_xy, saving_dir_train)          
        shutil.copy(file_yz, saving_dir_train)    
        shutil.copy(file_xz, saving_dir_train)
    else:
        missed_pdbs_train.append(pdb_name)
missed_pdbs_test = []
for pdb_name in test_pdbs:
    fname = ''.join([source_results_dir,'/',pdb_name,'_xy.txt'])
    if os.path.isfile(fname): 
        file_xy = ''.join([pdb_name,'_xy.txt'])
        file_yz = ''.join([pdb_name,'_yz.txt'])
        file_xz = ''.join([pdb_name,'_xz.txt'])
        shutil.copy(file_xy, saving_dir_test)          
        shutil.copy(file_yz, saving_dir_test)    
        shutil.copy(file_xz, saving_dir_test)
    else:
        missed_pdbs_test.append(pdb_name)
#%save the created randomised training unigene pdb ids as pikle files
os.chdir("/")
os.chdir(saving_pik)
pickle.dump(OG_whole_training_unigenes, open( ''.join([name,"_train_uni_gene.p"]), "wb" ))
pickle.dump(OG_whole_training_pdb_ids, open( ''.join([name,"_train_pdb_ids.p"]), "wb" ) )  
pickle.dump(OG_whole_testing_unigenes, open( ''.join([name,"_test_unoverlap_uni_gene.p"]), "wb" ))
pickle.dump(OG_whole_testing_pdb_ids, open( ''.join([name,"_test_unoverlap_pdb_ids.p"]), "wb" ) )  
print(name, " created :)")