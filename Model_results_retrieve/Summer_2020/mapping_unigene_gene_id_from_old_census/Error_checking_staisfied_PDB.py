# -*- coding: utf-8 -*-
"""
Created on %23-May-2020 at 1.41p.m

@author: %A.Nishanth C00294860
"""
import pickle
import os
from copy import deepcopy


#%% go through the older PDB details and find out which PDBs has to be placed
'''
Go through the previous train test lists and create the newly needed Train, Test PDB seperately
'''
main_dir='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/V_91_train_test_pickles'
os.chdir('/')
os.chdir(main_dir)
test_list_ids=pickle.load(open("test_list_ids.p", "rb"))  
train_list_ids=pickle.load(open("train_list_ids.p",  "rb"))  
ONGO_overlapped_gene_PDBs=pickle.load(open("ONGO_overlapped_gene_PDBs.p",  "rb"))  
TSG_overlapped_gene_PDBs=pickle.load(open("TSG_overlapped_gene_PDBs.p",  "rb"))  

#%%

old_main_dir='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/'
os.chdir('/')
os.chdir(old_main_dir)
test_list_ids_old=pickle.load(open("test_list_ids.p", "rb"))  
train_list_ids_old=pickle.load(open("train_list_ids.p",  "rb"))
overlapped_PDBs_ONGO_old=pickle.load(open("overlapped_PDBs_ONGO.p",  "rb"))
overlapped_PDBs_TSG_old=pickle.load(open("overlapped_PDBs_TSG.p",  "rb"))

#%%
test_PDB_needed=[]
for pdb in test_list_ids:
    if (pdb not in test_list_ids_old):# and (pdb not in train_list_ids_old):
        if pdb  in train_list_ids_old:
            print("Problem in unigene GeneID mapping of train set to test set")
            print('Go to mapping _dirr: check under this for loop commented')
            print('map again V_91_gene_list_save vin1')
        else:
            test_PDB_needed.append(pdb)
            
#mapping _dir=' C:\Users\nishy\Documents\Projects_UL\Continues BIBM\a----------model_results_retrieve\Spring_2020\T_1_and_T_2\mapping_unigene_gene_id_from_old_census'

train_PDB_needed=[]
for pdb in train_list_ids:
    if pdb not in train_list_ids_old:
        train_PDB_needed.append(pdb)
        
        
#%%
''' Go through the overlap files and problemtric categorised Genes and find the fifference between old census mapping 
Traing data and Testing mapping data        '''
def count_overlapped_PDBs_old_fell_by_geen_overlap(overlapped_PDBs_old,overlapped_gene_PDBs):
    count=0
    for pdb in overlapped_PDBs_old:
        if pdb in overlapped_gene_PDBs:
            count=count+1
    return count

og_ovre_count=count_overlapped_PDBs_old_fell_by_geen_overlap(overlapped_PDBs_ONGO_old,ONGO_overlapped_gene_PDBs)
tsg_ovre_count=count_overlapped_PDBs_old_fell_by_geen_overlap(overlapped_PDBs_TSG_old,TSG_overlapped_gene_PDBs)
#%% 
'''
To find out the missing PDBs of OLD to New

OR Error finding

'''
overlapped_gene_PDB=TSG_overlapped_gene_PDBs+ONGO_overlapped_gene_PDBs
list_ids=train_list_ids+test_list_ids
list_ids_old=train_list_ids_old+test_list_ids_old
missing=[]
coununt_in_overlap=0
for pdb in list_ids_old:
    if pdb not in list_ids:
        if pdb not in overlapped_gene_PDB:
            missing.append(pdb)
        else:
            coununt_in_overlap=coununt_in_overlap+1
'''
missing=[]
for pdb in train_list_ids_old:
    if pdb not in train_list_ids:
        if pdb not in overlapped_gene_PDB:
            missing.append(pdb)
            
'''
#%%
working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"
#% Gothrough the T_1 and create the results
Tier=1
amino_acids=21
name='TSG'

working_dir=''.join([working_dir_part,'Tier_',str(Tier),'_pdb_pikles_length_gene/'])
os.chdir('/')
os.chdir(working_dir)
overlapped_PDBs_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
overlapped_PDBs_TSG = pickle.load(open('overlapped_PDBs_TSG.p', "rb"))
os.chdir('/')
os.chdir(working_dir)
whole_PDBs = pickle.load(open(''.join([name,"_PDBs.p"]), "rb" ) )
All_load_dir ='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene/'
os.chdir('/')
os.chdir(All_load_dir)
SITE_satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
gene_ids = pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb")) 
for i in range(0,len(gene_ids)):
    count=0
    for pdb in SITE_satisfied_PDBs[i]:
        if ''.join([pdb,'.npy']) in missing:
            count=count+1
    if count>0:
        print(gene_ids[i])
count=0        
overlap_all=overlapped_PDBs_TSG+overlapped_PDBs_ONGO
for pdb in missing:
    if pdb[0:4] in overlap_all:
        count=count+1
    else:
        print(pdb[0:4])
        
        
if count==len(missing)-1:
    print("everythink is perfect due to 4OTW removed due to ONGO in new mapping")

#%%
overlapped_PDBs_old=  list(set(overlapped_PDBs_ONGO_old+overlapped_PDBs_TSG_old))
old_all=list(set(train_list_ids_old+overlapped_PDBs_old))
train_list_ids_old=list(set(train_list_ids_old))

new_all=list(set(train_list_ids+overlapped_gene_PDB+overlap_all))

#%%

'''
How satisfied PDBs created
_satisfied_PDBs

The answer can be found in 
C:\Users\nishy\Documents\Projects_UL\Continues BIBM\a----------PDB_Gene_Mapping\spring_2020\T1_AND_T2_rest\weight_according_to_genes
create_data_labels_splited_T_1.py: go through the ONGO,... PDBs and find the overlapping PDBs

C:\Users\nishy\Documents\Projects_UL\Continues BIBM\a----------model_results_retrieve\Spring_2020\T_1_and_T_2\mapping_unigene_gene_id_from_old_census
V_91_gene_list_save: using the older map information to map the gene ids.

 C:\Users\nishy\Documents\Projects_UL\Continues BIBM\a----------model_results_retrieve\Summer_2020
 to create the problemtric genes due to higheroverliap
 
 Finding_higher_overlap_genes_between_ONGO_and_TSG
'''
threshold_length=81
#working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"
##% Gothrough the T_1 and create the results
#Tier=1
#amino_acids=21
#
#working_dir=''.join([working_dir_part,'Tier_',str(Tier),'_pdb_pikles_length_gene/'])
#saving_dir=''.join([working_dir_part,'weights_assign_methods/used_Fully_clean_models/T_',str(Tier),'/SITE/before_thresh/',str(amino_acids),'_aminoacid/'])
#            
#name='TSG'
# # first load the SITE_MUST
#train_test_site_sat_dir=''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/SITE_must_Train_test_details/before_thresh/',str(amino_acids),'_aminoacid'])
#
#
##Inorder to use the training and testing data from 2018 census to model
#os.chdir('/')
#os.chdir(train_test_site_sat_dir)
#temp_11_pds = pickle.load(open('train_list_ids.p', "rb" ) )
#temp_22_pds = pickle.load(open('test_list_ids.p', "rb" ) )
#temp_1_pds=[]                    
#for pdbs in temp_11_pds:
#    temp_1_pds.append(pdbs[0:-4])
#temp_2_pds=[]                    
#for pdbs in temp_22_pds:
#    temp_2_pds.append(pdbs[0:-4])   
#    
#SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
#os.chdir('/')
#os.chdir(SITE_dir)
#SITE_satisfied_temp= pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))  
## since both 21 amino acids and 20 amino acids not-satisfied in have same PDBs
#not_satisfied_model=pickle.load(open('not_satisfied_model.p', "rb"))  #the PDB data unable to convert to supported quarter model formation 
#
#SITE_satisfied_all=list(set(temp_1_pds+temp_2_pds+SITE_satisfied_temp))
# 
## first load the files needed
#os.chdir('/')
#os.chdir(working_dir)
#whole_PDBs = pickle.load(open(''.join([name,"_PDBs.p"]), "rb" ) )
##        print(len(whole_PDBs))      
##        print(len(SITE_satisfied_all))
#
#site_sat_flatten=[]
#if Tier==1 and (name=='ONGO' or name=='TSG' or name=='Fusion'):
#        overlapped_PDBs_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
#        overlapped_PDBs_TSG = pickle.load(open('overlapped_PDBs_TSG.p', "rb"))
#        #to avoid not staified or overlap
#        overlap=list(set(overlapped_PDBs_ONGO+overlapped_PDBs_TSG+not_satisfied_model))
##                print('overlap: ',len(overlap))
#
#        for pdbs in whole_PDBs:
#            if pdbs in SITE_satisfied_all:
#                if pdbs not in overlap:
#                    site_sat_flatten.append(pdbs) 
#else:
#     for pdbs in whole_PDBs:
#        if pdbs in SITE_satisfied_all:
#            if pdbs not in not_satisfied_model:
#                site_sat_flatten.append(pdbs) 
#                
#                
#Ids_info_all = pickle.load(open( ''.join([name,"_Ids_info_all.p"]), "rb" ) )
#uni_gene = pickle.load(open( ''.join([name,"_Gene_ID.p"]), "rb" ) )
#
#PDBs_sel=[]
#start_end_position = []
#for ids in Ids_info_all:
#    PDB_t =[]
#    start_end_position_t =[]
#    for id_t in ids:
#        if len(id_t) == 3:
#            for j in range(0,len(id_t[2])):
#                if id_t[2][j]=="-":
#                   if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > threshold_length:
#                       if id_t[0] in site_sat_flatten:
#                           PDB_t.append(deepcopy(id_t[0]))
#                           start_end_position_t.append([int(id_t[2][0:j]),int(id_t[2][j+1:len(id_t[2])])])
##                               else:
##                                   print("Not_satisfied: ",id_t[0])
#
#    PDBs_sel.append(deepcopy(PDB_t))
#    start_end_position.append(deepcopy(start_end_position_t))
#
##%  after threshold satisfication tested the select gene and PDB details
#fin_PDB_ids = [] # finalised_PDB_ids 
#fin_PDB_ids_start_end_position = [] # finalised_PDB_ids_start_end_position
#fin_uni_gene = [] # finalised_uni_gene 
#for i in range(0,len(PDBs_sel)): 
#    if len(PDBs_sel[i]) > 0:   
#        fin_PDB_ids.append(PDBs_sel[i])
#        fin_PDB_ids_start_end_position.append(start_end_position[i])
#        fin_uni_gene.append(uni_gene[i])  
#        