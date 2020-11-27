# -*- coding: utf-8 -*-
"""
Created on %12-Jun-2020 at 3.49 p.m

@author: %A.Nishanth C00294860

Go through the higher overlapping genes and after processed(those primary structures removed); findout howmany PDBs overlapping

"""

import os
import pickle
from copy import deepcopy

num_amino_acids=21
SITE=True
print("warning the stst only calculated for Soft threhold only")

if SITE and num_amino_acids==21:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid'
elif SITE and num_amino_acids==20:
    main_directory='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid'

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
    
    unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
    os.chdir('/')
    os.chdir(unigene_load_dir)
    train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
    test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb"))    
    SITE_unigene_details_all=sum(SITE_unigene_details,[])
    Thres_sat_pdb_list_all=sum(Thres_sat_pdb_list,[])
    for pdb_id in SITE_unigene_details_all:
        if pdb_id not in Thres_sat_pdb_list_all:
            print('problemetric: ',pdb_id)
    train_unigene_details=[]       
    test_unigene_details=[]
    train_pdb_ids_details=[]
    test_pdb_ids_details=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] in clean_unigene_details:
            if Thres_sat_gene_list[i] not in gene_test_details:
                if len(SITE_unigene_details[i])>0:
                    train_pdb_ids_details.append(SITE_unigene_details[i])
                    train_unigene_details.append(Thres_sat_gene_list[i])
            else:
                if len(SITE_unigene_details[i])>0:
                    test_pdb_ids_details.append(SITE_unigene_details[i])
                    test_unigene_details.append(Thres_sat_gene_list[i])
  

#    not_there=[]
#    pdbs_left_due_to_process=[]
##    
#    train_new_processed_PDB_list=deepcopy(train_pdb_ids_details)
#    test_new_processed_pdb_list=deepcopy(test_pdb_ids_details)
#    del_train_list=[]
#    for gene in train_unigene_details:
#        if gene not in train_new_gene_list:
#            if gene not in test_new_gene_list:
#                print(gene)
#                for i in range(0,len(train_unigene_details)):
#                    if gene==train_unigene_details[i]:
#                        print(name,' Train earlier: ',gene)
#                        not_there.append([gene,deepcopy(train_pdb_ids_details[i])])   
#                        pdbs_left_due_to_process.append(deepcopy(train_pdb_ids_details[i]))
#                        del_train_list.append(i)
#    del_test_list=[]                  
#    for gene in test_unigene_details:
#        if gene not in train_new_gene_list:
#            if gene not in test_new_gene_list:
#                for i in range(0,len(test_unigene_details)):
#                    if gene==test_unigene_details[i]:
#                        print(name,' Test earlier: ',gene)
#                        not_there.append([gene,deepcopy(test_pdb_ids_details[i])])   
#                        pdbs_left_due_to_process.append(deepcopy(test_pdb_ids_details[i]))
#                        del_test_list.append(i)
#                        
#    del_train_list.sort() 
#    print(del_train_list)
#    del_test_list.sort() 
#    print(del_test_list)
#
#    while len(del_train_list)>0:
#        rem=del_train_list.pop()
#        print(train_unigene_details[rem],"  is removing")
#        print(" Number of PDBs in the group: ",len(train_new_processed_PDB_list[rem]))
#        del train_new_processed_PDB_list[rem]
#        
#    while len(del_test_list)>0:
#        del test_new_processed_pdb_list[del_test_list.pop(rem)]
##    del test_new_processed_pdb_list[i]
#
#    pdbs_left_due_to_process_all=list(set(sum(pdbs_left_due_to_process,[])))
#    print(name,' length pdbs_left_due_to_process_all: ',len(pdbs_left_due_to_process_all))
#    print("overlap all: ", len(overlaped_all))
#    still_needed=[]
#    for pdb in overlap_comp:
#        if pdb in pdbs_left_due_to_process_all:
#            still_needed.append(pdb)
#            
#    for pdb in pdbs_left_due_to_process_all:
#        if pdb in overlap_comp:
#            still_needed.append(pdb)
#    print(name, " lengh still_needed: ",len(still_needed))
#    
#    train_pdb_ids_details_all=list(set(sum(train_pdb_ids_details,[])))
#    test_pdb_ids_details_all=list(set(sum(test_pdb_ids_details,[])))
#    count=0
#    for pdb in overlaped_all:
#        if pdb in train_pdb_ids_details_all:
#            count=count+1
#    print("Overlapped PDBs occured Train: ",count)
#    
#    count=0
#    for pdb in overlaped_all:
#        if pdb in test_pdb_ids_details_all:
#            count=count+1
#    print("Overlapped PDBs occured Test: ",count)
    print(" ")
    print(name," bef train length: ",len(list(set(sum(train_pdb_ids_details,[])))))
    print(name," bef test length: ",len(list(set(sum(test_pdb_ids_details,[])))))
#
#    print(name," processed train length: ",len(list(set(sum(train_new_processed_PDB_list,[])))))
#    print(name," processed test length: ",len(list(set(sum(test_new_processed_pdb_list,[])))))
#    print(" ")

#    return not_there#,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details
    return train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details
#%%
name = "TSG"
train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
train_pdb_ids_details_tsg_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_tsg_all=list(set(sum(test_pdb_ids_details,[])))
pdbs_tsg=train_pdb_ids_details_tsg_all+test_pdb_ids_details_tsg_all
name = "Fusion"
train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
train_pdb_ids_details_Fusion_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_Fusion_all=list(set(sum(test_pdb_ids_details,[])))
fusion_all=train_pdb_ids_details_Fusion_all+test_pdb_ids_details_Fusion_all

name = "ONGO"
train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
train_pdb_ids_details_og_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_og_all=list(set(sum(test_pdb_ids_details,[])))
og_all=train_pdb_ids_details_og_all+test_pdb_ids_details_og_all
#overlaped= og_all+pdbs_tsg

#name = "TSG"
#not_there_tsg = chk_train_PDBs_list(name,SITE)
#
#name = "Fusion"
#not_there_fusion = chk_train_PDBs_list(name,SITE)
#name = "ONGO"
#not_there= chk_train_PDBs_list(name,SITE)

#%%
#name = "TSG"
#not_there_tsg,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
#train_pdb_ids_details_tsg_all=list(set(sum(train_pdb_ids_details,[])))
#test_pdb_ids_details_tsg_all=list(set(sum(test_pdb_ids_details,[])))
#pdbs_tsg=train_pdb_ids_details_tsg_all+test_pdb_ids_details_tsg_all
#name = "Fusion"
#not_there_fusion,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
#train_pdb_ids_details_Fusion_all=list(set(sum(train_pdb_ids_details,[])))
#test_pdb_ids_details_Fusion_all=list(set(sum(test_pdb_ids_details,[])))
#fusion_all=train_pdb_ids_details_Fusion_all+test_pdb_ids_details_Fusion_all
#
#name = "ONGO"
#not_there,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details = chk_train_PDBs_list(name,SITE)
#train_pdb_ids_details_og_all=list(set(sum(train_pdb_ids_details,[])))
#test_pdb_ids_details_og_all=list(set(sum(test_pdb_ids_details,[])))
#og_all=train_pdb_ids_details_og_all+test_pdb_ids_details_og_all
##overlaped= og_all+pdbs_tsg
#overlaped= og_all
#print("")
#count=0
