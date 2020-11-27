# -*- coding: utf-8 -*-
"""
Created on %12-Jun-2020 at 3.49 p.m

@author: %A.Nishanth C00294860

Go through the higher overlapping genes and after processed(those primary structures removed); findout howmany PDBs overlapping

"""

import os
import pickle
from copy import deepcopy

num_amino_acids=20
SITE=True
print("warning the stst only calculated for Soft threhold only")

if SITE and num_amino_acids==21:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid'
elif SITE and num_amino_acids==20:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid'

#%%
def fully_cleaned_count(main_directory):
    name="OG"
    os.chdir('/')
    os.chdir(main_directory)
    
    ONGO_pdbs= pickle.load(open(''.join([name,"_train_PDBs.p"]), "rb")) 
    name="TSG"
    TSG_pdbs= pickle.load(open(''.join([name,"_train_PDBs.p"]), "rb")) 
    name="Fusion"
    Fusion_pdbs= pickle.load(open(''.join([name,"_train_PDBs.p"]), "rb")) 
    
    overlapped_PDBs_OG= pickle.load(open("overlapped_PDBs_ONGO.p", "rb")) 
    overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb")) 
    overlap_all=overlapped_PDBs_OG+overlapped_PDBs_TSG
    for pdb in  ONGO_pdbs:
        if pdb in overlap_all:
            print(pdb, "in ONGO overlapped problem")
    for pdb in  TSG_pdbs:
        if pdb in overlap_all:
            print(pdb,"in TSG overlapped problem")
    for pdb in Fusion_pdbs:
        if pdb in overlap_all:
            print(pdb,"in Fusion overlapped problem")
    #% first create the data for 10-fold dataset
    test_labels_dic = pickle.load(open("test_labels_dic.p", "rb")) 
    test_list_ids= pickle.load(open("test_list_ids.p", "rb")) 
    
    for pdb in test_list_ids:
        if pdb in overlap_all:
            print("Test PDB in overalpping PDB:",pdb)
    OG_test_count=0
    TSG_test_count=0
    Fusion_test_count=0
    
    OG_all_count=len(ONGO_pdbs)
    TSG_all_count=len(TSG_pdbs)
    Fusion_all_count=len(Fusion_pdbs)
    negate_all=0
    for pdb in test_list_ids:
       if test_labels_dic[pdb]==0:
           OG_test_count=OG_test_count+1
           if pdb not in ONGO_pdbs:
               OG_all_count=OG_all_count+1
           else:
                negate_all=negate_all+1
       elif test_labels_dic[pdb]==1:
           TSG_test_count=TSG_test_count+1
           if pdb not in TSG_pdbs:
               TSG_all_count=TSG_all_count+1   
           else:
                negate_all=negate_all+1
       elif test_labels_dic[pdb]==2:
           Fusion_test_count=Fusion_test_count+1
           if pdb not in Fusion_pdbs:
               Fusion_all_count=Fusion_all_count+1  
           else:
                negate_all=negate_all+1
    all_pdbs = ONGO_pdbs+TSG_pdbs+Fusion_pdbs+test_list_ids
    fully_clean_count_dic={}
    fully_clean_count_dic['train_ONGO']=len(ONGO_pdbs)
    fully_clean_count_dic['train_TSG']=len(TSG_pdbs)
    fully_clean_count_dic['train_Fusion']=len(Fusion_pdbs)
    fully_clean_count_dic['test_ONGO']=OG_test_count
    fully_clean_count_dic['test_TSG']=TSG_test_count
    fully_clean_count_dic['test_Fusion']=Fusion_test_count
    fully_clean_count_dic['all']=len(list(set(all_pdbs)))
    fully_clean_count_dic['all_ONGO']=OG_all_count
    fully_clean_count_dic['all_TSG']=TSG_all_count
    fully_clean_count_dic['all_Fusion']=Fusion_all_count

    return fully_clean_count_dic,list(set(all_pdbs))

fully_clean_count_dic,clean_all_pdbs = fully_cleaned_count(main_directory)
#%% check actually howmany PDBs satisfied 81 threhold and left due to MSMS or surface threghold of surface issue

def chk_train_PDBs_list(name,SITE):
    
    
    os.chdir('/')
    os.chdir(main_directory)
    
    overlap_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p','rb'))
    overlap_TSG = pickle.load(open('overlapped_PDBs_TSG.p','rb'))
    overlap_comp=overlap_ONGO+overlap_TSG
    overlaped_all=[]
    for pdb in overlap_comp:
        overlaped_all.append(pdb[0:-4])
        
    loading_thresh_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
    os.chdir("/")
    os.chdir(loading_thresh_dir)
    if SITE:
        if name=='TSG':

    #        pdbs_not_software_must = pickle.load(open(''.join([name,"_pdbs_not_software_must.p"]), "rb" ))
            pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_must_gene.p"]), "rb")) 
#            pdbs_not_software_must_gene=[]
        else:
            pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_gene.p"]), "rb" ))
        print(name,': ',len(pdbs_not_software_must_gene))
#        for i in pdbs_not_software_must_gene:
#            print(i)
#        os.chdir('/')
#        os.chdir(saving_dir)    
#        train_pdb_ids_details= pickle.load(open(''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
#        train_unigene_details= pickle.load(open(''.join([name,"_train_unigene_details.p"]), "rb" ))
#        test_pdb_ids_details = pickle.load(open(''.join([name,"_test_pdb_ids_details.p"]), "rb" ))
#        test_unigene_details = pickle.load(open(''.join([name,"_test_unigene_details.p"]), "rb" )) 
         
    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    
    os.chdir("/")
    os.chdir(site_dir)
    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))

    SITE_unigene_details_all=sum(SITE_unigene_details,[])
    Thres_sat_pdb_list_all=sum(Thres_sat_pdb_list,[])
    for pdb_id in SITE_unigene_details_all:
        if pdb_id not in Thres_sat_pdb_list_all:
            print('problemetric: ',pdb_id)
            

    if num_amino_acids==21:
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
        test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
    elif num_amino_acids==20:
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/20_amino_acids_fully_clean_SITE'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_20.p"]), "rb")) 
        test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_20.p"]), "rb")) 
    
    not_there=[]
    train_unigene_details=[]       
    test_unigene_details=[]
    train_pdb_ids_details=[]
    test_pdb_ids_details=[]
    left_PDbs=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] in train_new_gene_list:
            if len(SITE_unigene_details[i])>0:
                    train_pdb_ids_details.append(SITE_unigene_details[i])
                    train_unigene_details.append(Thres_sat_gene_list[i])
        elif Thres_sat_gene_list[i] in test_new_gene_list:
                if len(SITE_unigene_details[i])>0:
                    test_pdb_ids_details.append(SITE_unigene_details[i])
                    test_unigene_details.append(Thres_sat_gene_list[i])
        else:
            if len(SITE_unigene_details[i])>0:
                not_there.append([Thres_sat_gene_list[i],deepcopy(SITE_unigene_details[i])])
                left_PDbs.append(deepcopy(SITE_unigene_details[i]))
   
    left_PDbs_all=list(set(sum(left_PDbs,[])))
    
    all_processed_train= list(set(sum(train_pdb_ids_details,[])))
    all_processed_test = list(set(sum(test_pdb_ids_details,[])))
    print(name," processed train PDB length: ",len(all_processed_train))
    print(name," processed test PDB length: ",len(all_processed_test))


    all_pdbs=list(set(all_processed_train+all_processed_test+left_PDbs_all))
    print(name," processed overall PDB length: ",len(list(set(all_processed_train+all_processed_test))))
    print(name," without processed all PDB length: ",len(all_pdbs))

    print(" ")
    print(name," processed train gene length: ",len(train_pdb_ids_details))
    print(name," processed test gene length: ",len(test_pdb_ids_details))
    print(name," processed overall gene length: ",len(train_pdb_ids_details+test_pdb_ids_details))
    print(name," without processed all gene length: ",len(train_pdb_ids_details)+len(test_pdb_ids_details)+len(not_there))

    print(" ")
    print(" ")
    return left_PDbs_all,not_there,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details
#    return train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details
#%%
    
name = "ONGO"
left_PDbs_OG,not_there,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details_og,test_unigene_details_og,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
train_pdb_ids_details_og_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_og_all=list(set(sum(test_pdb_ids_details_og,[])))
og_all=train_pdb_ids_details_og_all+test_pdb_ids_details_og_all

name = "TSG"
left_PDbs_TSG,not_there_tsg,train_unigene_details_tsg,train_pdb_ids_details_tsg,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
train_pdb_ids_details_tsg_all=list(set(sum(train_pdb_ids_details_tsg,[])))
test_pdb_ids_details_tsg_all=list(set(sum(test_pdb_ids_details,[])))
pdbs_tsg=train_pdb_ids_details_tsg_all+test_pdb_ids_details_tsg_all
name = "Fusion"
left_PDbs_Fusion,not_there_fusion,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
train_pdb_ids_details_Fusion_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_Fusion_all=list(set(sum(test_pdb_ids_details,[])))
fusion_all=train_pdb_ids_details_Fusion_all+test_pdb_ids_details_Fusion_all

os.chdir('/')
os.chdir(main_directory)
overlap_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p','rb'))
overlap_TSG = pickle.load(open('overlapped_PDBs_TSG.p','rb'))
overlap_comp=overlap_ONGO+overlap_TSG
overlaped_all=[]
for pdb in overlap_comp:
    overlaped_all.append(pdb[0:-4])
    
left_PDBS_all=left_PDbs_OG+left_PDbs_TSG+left_PDbs_Fusion
for pdb in overlaped_all:
    if pdb not in left_PDBS_all:
        print("PDB come from some other PDBs: ",pdb)
all_preprocess=list(set(og_all+pdbs_tsg+fusion_all))
all_bef_process=list(set(left_PDBS_all+all_preprocess))
#%%
preprocess_prob=[]
for pdb in all_preprocess:
    if ''.join([pdb,'.npy']) not in clean_all_pdbs:
        preprocess_prob.append(pdb)
#%%
'''
the ONGO problemtric PDBs mentioned in the table report 

since PDB file “4MDQ” not able produce results through MSMS tool.
PDBs “3GT8” and “3LZB” has unknown amino acids so those left and PDB “721P” 
has no single Cα surface atom by the threshold condition as mentioned in section 1.3. 
'''
train_all= train_pdb_ids_details_og_all+train_pdb_ids_details_tsg_all+train_pdb_ids_details_Fusion_all
test_all=test_pdb_ids_details_og_all+test_pdb_ids_details_tsg_all+test_pdb_ids_details_Fusion_all

for pdb in preprocess_prob:
    if pdb in og_all:
        print(pdb)
        if pdb in train_all:
            print("in train")
        if pdb in test_all:
            print("in test")
            for i in range(0,len(test_pdb_ids_details_og)):
                if pdb in test_pdb_ids_details_og[i]:
                    print(i)
                    print(test_unigene_details_og[i])
#%%
'''
TSG missing sort

5C08 and 5C0B has size issue
and 2H26 has unknown aminoacid problem
'''        
not_in_overlap_chk=[]  
overlap_count=0  
for pdb in preprocess_prob:
    if pdb in pdbs_tsg:
        if pdb not in overlaped_all:
            not_in_overlap_chk.append(pdb)
        else:
            overlap_count=overlap_count+1
print("overlap in TSG count: ",overlap_count)
for pdb in not_in_overlap_chk:
    print(pdb)
    if pdb in train_all:
        print("in train")
    if pdb in test_all:
        print("in test")

loading_dir_threshold = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
#         
os.chdir("/")
os.chdir(loading_dir_threshold)  
tsg_soft_prob = pickle.load(open('TSG_pdbs_not_software_must_exp_type.p','rb'))  
#%%
'''
Fusion missing sort


'''        
not_in_overlap_chk=[]  
overlap_count=0  
for pdb in preprocess_prob:
    if pdb in fusion_all:
        if pdb not in overlaped_all:
            not_in_overlap_chk.append(pdb)
        else:
            overlap_count=overlap_count+1
print("overlap in Fusion count: ",overlap_count)
for pdb in not_in_overlap_chk:
    print(pdb)
    if pdb in train_all:
        print("in train")
    if pdb in test_all:
        print("in test")
