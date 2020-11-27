# -*- coding: utf-8 -*-
"""
Created on %08-Nov-2018

@author: %A.Nishanth C00294860
"""
import pickle
import os

UniGenes_for_train = ['Hs.719495', 'Hs.479756','Hs.706627','Hs.726012']

traing_testing_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/Train_test_select_based_SITE_info"
name = "ONGO"
#%% then create the whole set
os.chdir("/")
os.chdir(traing_testing_pdb_dir)
whole_training_unigene_2 = pickle.load(open(''.join([name,"_whole_train_unigene.p"]), "rb" ) ) 
whole_testing_unigene_2 = pickle.load(open(''.join([name,"_whole_test_unigene.p"]), "rb" ) ) 
#%%
os.chdir("/")
os.chdir(traing_testing_pdb_dir)
# training data pickle
whole_training_unigene = pickle.load(open(''.join([name,"_whole_train_unigene.p"]), "rb" ) ) 
# testing data pickle
whole_testing_unigene = pickle.load(open(''.join([name,"_whole_test_unigene.p"]), "rb" ) ) 

ongo_uni_gene_pdb_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/"
os.chdir("/")
os.chdir(ongo_uni_gene_pdb_dir)
ONGO_unigene = pickle.load(open(''.join([name,"_thres_sat_gene_list.p"]), "rb" ) )
ONGO_unigene_PDB = pickle.load(open(''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ) )
#%% then goahead and swap the selected testing_unigene to training unigene
from random import shuffle
import copy

#% create random number 
indexes = list(range(0,len(whole_training_unigene)))
#then randomize the indexes
shuffle(indexes)
select_randomly_4_unigene_from_train = []
for i in range(0,4):
    select_randomly_4_unigene_from_train.append(copy.deepcopy(whole_training_unigene[indexes[i]]))
    whole_training_unigene[indexes[i]] = UniGenes_for_train[i]
    print("Added: ", UniGenes_for_train[i])
for j in range(0,len(UniGenes_for_train)):  
    for i in range(0,len(whole_testing_unigene)):
        if whole_testing_unigene[i]==UniGenes_for_train[j]:
            print("UniGene in test ",whole_testing_unigene[i])
            whole_testing_unigene[i] = select_randomly_4_unigene_from_train[j]
            print(" has changed to ",whole_testing_unigene[i])
#%% then create the training pdb ids
whole_training_pdb_ids = []
for u in whole_training_unigene:
    for i in range(0,len(ONGO_unigene)):
        if ONGO_unigene[i] == u:
            whole_training_pdb_ids.append(ONGO_unigene_PDB[i])
whole_testing_pdb_ids= []
for u in whole_testing_unigene:
    for i in range(0,len(ONGO_unigene)):
        if ONGO_unigene[i] == u:
            whole_testing_pdb_ids.append(ONGO_unigene_PDB[i])    
#%% then save them as pikle files to retrieve later
import shutil
saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/ongo_again_2"
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
pickle.dump(whole_testing_unigene, open( ''.join([name,"_test_uni_gene.p"]), "wb" ))
pickle.dump(whole_testing_pdb_ids, open( ''.join([name,"_test_pdb_ids.p"]), "wb" ) )  
print(name, " created :)")
#%% 
import pickle
import os
name = "ONGO"
saving_pik = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/ongo_again"
os.chdir("/")
os.chdir(saving_pik)

whole_testing_unigene = pickle.load(open(''.join([name,"_test_uni_gene.p"]), "rb" ))
whole_testing_pdb_ids = pickle.load(open(''.join([name,"_test_pdb_ids.p"]), "rb" )) 
test_pdbs = sum(whole_testing_pdb_ids, [])