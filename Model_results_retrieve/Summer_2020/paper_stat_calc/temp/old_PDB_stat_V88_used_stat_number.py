# -*- coding: utf-8 -*-
"""
Created on %(20-May-2020) at 6.28pm

@author: %A.Nishanth C00294860 on 11-Jun-2020 at 10.31 A.m

This file load the pickles and findout the numberes for stat mentioned in paper
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
#%%
'''
Count the number of PDBs with overlap
'''

def chk_PDBs_with_overlapped(name,main_directory):
    Test_dir_part=''.join([main_directory,'/Test/',name])
    Trian_dir_part=''.join([main_directory,'/Train/',name])
    os.chdir('/')
    os.chdir(Trian_dir_part)
    files=os.listdir()
    for pdb in files:
        if  pdb[-3:]!='npy':
            print(pdb)
            os.chdir('/')
    os.chdir(Test_dir_part)
    files_t=os.listdir()
    for pdb in files_t:
        if  pdb[-3:]!='npy':
            print(pdb)
    return len(files),len(files_t)

def preprocessed_PDBs_with_over_lap(main_dir):
    train_ONGO,test_ONGO=chk_PDBs_with_overlapped("ONGO",main_directory)
    train_TSG,test_TSG=chk_PDBs_with_overlapped("TSG",main_directory)
    train_Fusion,test_Fusion=chk_PDBs_with_overlapped("Fusion",main_directory)
    processed_count_dic={}
    processed_count_dic['train_ONGO']=train_ONGO
    processed_count_dic['train_TSG']=train_TSG
    processed_count_dic['train_Fusion']=train_Fusion
    processed_count_dic['test_ONGO']=test_ONGO
    processed_count_dic['test_TSG']=test_TSG
    processed_count_dic['test_Fusion']=test_Fusion
    return processed_count_dic

processed_count_dic=preprocessed_PDBs_with_over_lap(main_directory)

##%% check actually howmany PDBs satisfied 81 threhold and left due to MSMS or surface threghold of surface issue
#
#def chk_train_PDBs_list(name,SITE):
#    loading_thresh_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
#    os.chdir("/")
#    os.chdir(loading_thresh_dir)
#    if SITE:
#        if name=='TSG':
#
#    #        pdbs_not_software_must = pickle.load(open(''.join([name,"_pdbs_not_software_must.p"]), "rb" ))
#            pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_must_gene.p"]), "rb")) 
##            pdbs_not_software_must_gene=[]
#        else:
#            pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_gene.p"]), "rb" ))
#        print(name,': ',len(pdbs_not_software_must_gene))
##        for i in pdbs_not_software_must_gene:
##            print(i)
##        os.chdir('/')
##        os.chdir(saving_dir)    
##        train_pdb_ids_details= pickle.load(open(''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
##        train_unigene_details= pickle.load(open(''.join([name,"_train_unigene_details.p"]), "rb" ))
##        test_pdb_ids_details = pickle.load(open(''.join([name,"_test_pdb_ids_details.p"]), "rb" ))
##        test_unigene_details = pickle.load(open(''.join([name,"_test_unigene_details.p"]), "rb" )) 
#         
#    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
#    
#    os.chdir("/")
#    os.chdir(site_dir)
#    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
#    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
#    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
#    
#    clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
#    os.chdir("/")
#    os.chdir(clean_pikle_dir)
#    if name=="ONGO":
##        gene_test_details = ['Hs.731652','Hs.37003','Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
#        gene_test_details = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
#        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
#    elif name =="TSG":
#        '''No Hs.517792'''
#        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
#        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
#        if pdbs_not_software_must_gene[0] in gene_test_details:
#            raise("chcek the TSG")
#    elif name =="Fusion":
#        gene_test_details = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
#        clean_unigene_details = Thres_sat_gene_list
#        
#    SITE_unigene_details_all=sum(SITE_unigene_details,[])
#    Thres_sat_pdb_list_all=sum(Thres_sat_pdb_list,[])
#    for pdb_id in SITE_unigene_details_all:
#        if pdb_id not in Thres_sat_pdb_list_all:
#            print('problemetric: ',pdb_id)
#    train_unigene_details=[]       
#    test_unigene_details=[]
#    train_pdb_ids_details=[]
#    test_pdb_ids_details=[]
#    for i in range(0,len(Thres_sat_gene_list)):
#        if Thres_sat_gene_list[i] in clean_unigene_details:
#            if Thres_sat_gene_list[i] not in gene_test_details:
#                if len(SITE_unigene_details[i])>0:
#                    train_pdb_ids_details.append(SITE_unigene_details[i])
#                    train_unigene_details.append(Thres_sat_gene_list[i])
#            else:
#                if len(SITE_unigene_details[i])>0:
#                    test_pdb_ids_details.append(SITE_unigene_details[i])
#                    test_unigene_details.append(Thres_sat_gene_list[i])
#  
#            
#    unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
#    os.chdir('/')
#    os.chdir(unigene_load_dir)
#    train_new_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
#    test_new_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
#    not_there=[]
#    for gene in train_unigene_details:
#        if gene not in train_new_gene_list:
#            if gene not in test_new_gene_list:
#                print(gene)
#                for i in range(0,len(train_unigene_details)):
#                    if gene==train_unigene_details[i]:
#                        print(name,' Train earlier: ',gene)
#                        not_there.append([gene,deepcopy(train_pdb_ids_details[i])])   
#                for i in range(0,len(test_unigene_details)):
#                    if gene==test_unigene_details[i]:
#                        print(name,' Test earlier: ',gene)
#                        not_there.append([gene,deepcopy(test_pdb_ids_details[i])])   
#    return not_there,train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details
#
##%%
#
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
##print("Number of PDBs in the group: ",len(not_there_fusion[0][1]))
##for pdb in not_there_fusion[0][1]:
#for pdb in not_there_tsg[0][1]:
#    if pdb in overlaped:
#        count=count+1
#print("Number of PDBs in the group: ",len(not_there_tsg[0][1]))
#print("Number of PDBs in the overlapped: ",count)
#
##%%
#'''
#Check these genes overlapped with ONGO and TSG 
#
#In cleaning process these genes are left in ONGO and TSG
#'''
#
##name="TSG"
##clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
##os.chdir("/")
##os.chdir(clean_pikle_dir)
##if name=="TSG":
###        gene_test_details = ['Hs.731652','Hs.37003','Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
##    clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
##    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
##    
##    os.chdir("/")
##    os.chdir(site_dir)
##    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
##    for gene in Thres_sat_gene_list:
##        if gene not in clean_unigene_details:
##            print(gene)
##%% 
'''
checking the overlapped with the seleted genes and their count
'''
def chk_finalised_PDBs_list(name,SITE,num_amino_acids):
#    loading_thresh_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
#    os.chdir("/")
#    os.chdir(loading_thresh_dir)
#    if SITE:
#        if name=='TSG':
#
#    #        pdbs_not_software_must = pickle.load(open(''.join([name,"_pdbs_not_software_must.p"]), "rb" ))
#            pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_must_gene.p"]), "rb")) 
#        else:
#            pdbs_not_software_must_gene = pickle.load(open(''.join([name,"_pdbs_not_software_gene.p"]), "rb" ))
#        print(name,': ',len(pdbs_not_software_must_gene))
##        for i in pdbs_not_software_must_gene:
##            print(i)
##        os.chdir('/')
##        os.chdir(saving_dir)    
##        train_pdb_ids_details= pickle.load(open(''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
##        train_unigene_details= pickle.load(open(''.join([name,"_train_unigene_details.p"]), "rb" ))
##        test_pdb_ids_details = pickle.load(open(''.join([name,"_test_pdb_ids_details.p"]), "rb" ))
##        test_unigene_details = pickle.load(open(''.join([name,"_test_unigene_details.p"]), "rb" )) 
#         
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
        
    train_unigene_details=[]       
    test_unigene_details=[]
    train_pdb_ids_details=[]
    test_pdb_ids_details=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] in train_new_gene_list:
            if Thres_sat_gene_list[i] not in test_new_gene_list:
                if len(SITE_unigene_details[i])>0:
                    train_pdb_ids_details.append(SITE_unigene_details[i])
                    train_unigene_details.append(Thres_sat_gene_list[i])
        elif Thres_sat_gene_list[i] in test_new_gene_list:
                if len(SITE_unigene_details[i])>0:
                    test_pdb_ids_details.append(SITE_unigene_details[i])
                    test_unigene_details.append(Thres_sat_gene_list[i])

    return train_unigene_details,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details,test_pdb_ids_details

name = "TSG"
train_unigene_details_tsg,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details_tsg,test_pdb_ids_details = chk_finalised_PDBs_list(name,SITE,num_amino_acids)
train_pdb_ids_details_tsg_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_tsg_all=list(set(sum(test_pdb_ids_details,[])))
name = "Fusion"
train_unigene_details_fusion,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details_fusion,test_pdb_ids_details = chk_finalised_PDBs_list(name,SITE,num_amino_acids)
train_pdb_ids_details_Fusion_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_Fusion_all=list(set(sum(test_pdb_ids_details,[])))
name = "ONGO"
train_unigene_details_og,train_pdb_ids_details,test_pdb_ids_details,test_unigene_details_og,test_pdb_ids_details = chk_finalised_PDBs_list(name,SITE,num_amino_acids)
train_pdb_ids_details_og_all=list(set(sum(train_pdb_ids_details,[])))
test_pdb_ids_details_og_all=list(set(sum(test_pdb_ids_details,[])))
#%%


