# -*- coding: utf-8 -*-
"""
Created on %(13-Sep-2018) 16.07Pm

@author: %A.Nishanth C00294860

Tier_1 used for craeting the training data"""
import os
import pickle
import copy
import math
# to load the sitre information
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

#%% load the genedetails to select the PDB data 
# count the number of PDBs for each gene
name = "ONGO"

os.chdir('/')
os.chdir(site_dir)

def intialised_part(name):
    """
    This function load the pikle information about SITE satisfied ids 
    and the threshold staisfied information
    """
    fin_PDB_ids = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))    
    fin_uni_gene = pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))  
    gene_sat = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]),  "rb"))                               
                
    # then count the occurance of number of PDBs in this time
    site_sat_occurance_counter = []
    for k in gene_sat:
        site_sat_occurance_counter.append(len(k))
    thrs_sat_occurance_counter = []
    for k in fin_PDB_ids:
        thrs_sat_occurance_counter.append(len(k))
    return  fin_PDB_ids, fin_uni_gene, gene_sat, site_sat_occurance_counter, thrs_sat_occurance_counter 

#%% load the information about ONGO, TSG and Fusion data
ONGO_fin_PDB_ids, ONGO_fin_uni_gene,  ONGO_gene_sat, ONGO_site_sat_occurance_counter,  ONGO_thrs_sat_occurance_counter = intialised_part("ONGO") 
#TSG_fin_PDB_ids, TSG_fin_uni_gene,  TSG_gene_sat, TSG_site_sat_occurance_counter,  TSG_thrs_sat_occurance_counter = intialised_part("TSG") 
#Fusion_fin_PDB_ids, Fusion_fin_uni_gene,  Fusion_gene_sat, Fusion_site_sat_occurance_counter,  Fusion_thrs_sat_occurance_counter = intialised_part("Fusion")      
site_sat_occurance_counter =   ONGO_site_sat_occurance_counter  
fin_uni_gene = ONGO_fin_uni_gene
gene_sat = ONGO_gene_sat
fin_PDB_ids = ONGO_fin_PDB_ids
 #%%
def frequent_counter(given_list,selector):
    """
    selector: choose the return type
    Make the unique elements from the given list
    Then
        count howmany times each unique element occurs accordingly
    
    """
    Unique_given_list =  list(set(given_list))
    freq = []
    index = []
    for i in range(0, len(Unique_given_list)):
        count = 0
        temp = []
        for j in range(0,len(given_list)):
            if Unique_given_list[i] == given_list[j]:
                  count = count + 1
                  temp.append(j)
        freq.append(count)
        index.append(temp)
    if selector == 0:
       return Unique_given_list, freq, index
    else:
        return Unique_given_list, freq
    
def indexes_for_training_and_testing(site_sat_occurance_counter):
    """
    This funstion is a helper function for the function "training_data_stat_claculate"
    
    This function returns the indexes for training and testing using
    sliding window size of 4 with 1 for testing and rest(3) for training
    
    Group | cumul | elements
          | ative | exist in 
          | count | that group
                  
    1       11         1 [1]
    2                  1   [1]
    3       7          3     [1 1 1] 
    4       3          6            [1 1 1 1 1 1]
            window       [--4---1][--4---2][-3-3]
       
   First case(from the window [--4---1])
       group_1 choosen for test_data
       group_2 ,group_3 and  group_3 are choosen for choosen for train_data
       
   Second case(from the window [--4---2])
       group_3 choosen for test_data
       group_4 ,group_4 and  group_4 are choosen for choosen for train_data
       
   Third case(from the window [-3-3])
       since only three left after window 2 is gone, so choose them all for testing
           group_4 ,group_4 and  group_4 are choosen for choosen for train_data
                
    """
    Unique_site_sat_occurance_counter =  list(set(site_sat_occurance_counter))
    # then organize them in the assending order
    # to make sure the higher occurance left it goes for the training
    Unique_site_sat_occurance_counter_assending_key =   [i[0] for i in sorted(enumerate(Unique_site_sat_occurance_counter), key=lambda x:x[1])]
    
    # then the count of each element of UniGene_unique_assending  
    # with the unique_elements_in Unique_site_sat_occurance_counter check how frequent they occurs
    Unique_site_sat_occurance_counter, UniGene_unique_freq, UniGene_index = frequent_counter(site_sat_occurance_counter,0)

    training_data =[]
    testing_data =[]
    if Unique_site_sat_occurance_counter[Unique_site_sat_occurance_counter_assending_key[0]] == 0:
        m = 1 # index of UniGene to avoid 0 numeber of SITE occurance
        cumulative_count = sum(UniGene_unique_freq) -  UniGene_unique_freq[Unique_site_sat_occurance_counter_assending_key[0]]
    else:
        m=0
        cumulative_count = sum(UniGene_unique_freq) 
    carry = 0 # to hold the number of elements take for the next one   intial condition
    while cumulative_count > 0:
        i = Unique_site_sat_occurance_counter_assending_key[m]
        if carry < 0:
            carry = 0  
        for j in range(carry,UniGene_unique_freq[i]-3,4):
            testing_data.append(m) 
            for k in range(0,3):
                training_data.append(m)
        
        # to staisfy the ending condition
        cond_2 = True
        if cumulative_count < carry: 
            cond_2 = False
            if m+1 == len(Unique_site_sat_occurance_counter_assending_key):
                for j in range(0,cumulative_count):
                    training_data.append(m)
                break
            else:
                for j in range(0,UniGene_unique_freq[i]):
                    training_data.append(m)
                carry = carry - UniGene_unique_freq[i]
        if cond_2:
            new_count = UniGene_unique_freq[i]-carry    
            if new_count >= 4:# since 4 is the window size
                # inorder to add the remaining data for training
                # to add the front part left
                for j in range(0,carry):
                    training_data.append(m)
                # to add the end part left
                if cumulative_count < 4: 
                    if (new_count % 4 != 0):
                        c = list(range(UniGene_unique_freq[i] - (new_count % 4) ,UniGene_unique_freq[i]))
                        for k in c:
                            training_data.append(m)
                else:
                    if (new_count % 4 != 0):
                        c = list(range(UniGene_unique_freq[i] -(new_count % 4) ,UniGene_unique_freq[i]))
                        cond = True # to place the first element of the stack to the test data
                        carry = 4
                        for k in c:
                            if cond and (cumulative_count - UniGene_unique_freq[i])>0:
                                testing_data.append(m)
                                cond= False
                            else:
                                training_data.append(m)
                            carry = carry-1
            else: 
                # in this case consider about overlap front and end
                # to add the front part left
                if carry > 0 and UniGene_unique_freq[i]>0:
                    if UniGene_unique_freq[i]>carry:
                        for j in range(0,carry):
                            training_data.append(m)
                        testing_data.append(m)
                        COUNT = 0
                        for j in range(0, UniGene_unique_freq[i] -1 - carry):
                            training_data.append(m)
                            COUNT = COUNT +1
                            
                        carry = 4 - COUNT 
                    else:
                        for j in range(0,UniGene_unique_freq[i]):
                            training_data.append(m)
                        carry = carry - UniGene_unique_freq[i]
                elif  UniGene_unique_freq[i]>0:
                    carry = 4
                    testing_data.append(m)
                    for j in range(1,UniGene_unique_freq[i]):
                        training_data.append(m)
                        carry = carry-1
                    carry = carry - UniGene_unique_freq[i]
        cumulative_count = cumulative_count - UniGene_unique_freq[i]
        m =m + 1
    print("Actual                  : ", sum(UniGene_unique_freq) -  UniGene_unique_freq[Unique_site_sat_occurance_counter_assending_key[0]])
    print("GOT                     : ", len(training_data)+len(testing_data))
    print("Training_data_percentage: ",100*len(training_data)/(len(training_data)+len(testing_data)))
    return training_data,testing_data,  UniGene_index, Unique_site_sat_occurance_counter_assending_key
#%%

def index_chooses_training_testing(fin_uni_gene, fin_PDB_ids, training_data, gene_sat, testing_data, UniGene_index, Unique_site_sat_occurance_counter_assending_key):
    """
    When the training data is choosen from the UniGene list 
    Select the UniGene with the minimum number of pdbs_without SITE information
                                to maximum number of pdbs_without SITE information
     Eg: suppose 
            Gene_1: 10 Site_pdbs & 23 missing(SITE info) pdbs
            Gene_2: 10 Site_pdbs & 21 missing(SITE info) pdbs
            
            then the Gene_2 is selected for the training set first
            
    When the set is selecting for testing data
    Selection process is in the otherway
     Eg: suppose 
            Gene_1: 10 Site_pdbs & 23 missing(SITE info) pdbs
            Gene_2: 10 Site_pdbs & 21 missing(SITE info) pdbs
            
            then the Gene_1 is selected for the testing set first
    """
    # then select the training data from the index
    training_data_Uni_Gene_indexes =[]
    # count the frequent of occurance in the training data
    unique_training_data, training_data_unique_frequent = frequent_counter(training_data,1)
    #% count the number of pdb_s without SITE_info
    count_missing_SITE_pdbs = []
    for i in range(0,len(fin_uni_gene)):
        count_missing_SITE_pdbs.append(len(fin_PDB_ids[i])-len(gene_sat[i]))
    
    for i in range(0,len(unique_training_data)):
        # first choose the list we wanted  
        list_index_wanted = UniGene_index[Unique_site_sat_occurance_counter_assending_key[unique_training_data[i]]]
        missing_indexes_for_the_list_wanted = []
        for j in  list_index_wanted:
            missing_indexes_for_the_list_wanted.append(count_missing_SITE_pdbs[j])
        key_missing = [k[0] for k in sorted(enumerate(missing_indexes_for_the_list_wanted), key=lambda x:x[1])]    
        for j in  range(0,training_data_unique_frequent[i]):
            training_data_Uni_Gene_indexes.append(list_index_wanted[key_missing[j]])
    
    #% then select the testing data from the index
    testing_data_Uni_Gene_indexes = []
    # count the frequent of occurance in the testing data
    unique_testing_data, testing_data_unique_frequent = frequent_counter(testing_data,1)
    
    for i in range(0,len(unique_testing_data)):
        # first choose the list we wanted  
        list_index_wanted = UniGene_index[Unique_site_sat_occurance_counter_assending_key[unique_testing_data[i]]]
        missing_indexes_for_the_list_wanted = []
        for j in  list_index_wanted:
            missing_indexes_for_the_list_wanted.append(count_missing_SITE_pdbs[j])
        key_missing = [k[0] for k in sorted(enumerate(missing_indexes_for_the_list_wanted), key=lambda x:x[1])] 
        # for tesing choose the Genes from the backside way
        for j in  range(1,testing_data_unique_frequent[i]+1):
            testing_data_Uni_Gene_indexes.append(list_index_wanted[key_missing[-j]])
            
    # combine both traing and testing lists and check where missed
    """
    This is for checking purpos eto see evey think occurs correctly
    """
    all_chk =[]
    for i in testing_data_Uni_Gene_indexes:
        all_chk.append(i)
    for i in training_data_Uni_Gene_indexes:
        all_chk.append(i)
    all_chk_unique =  list(set(all_chk))
    
    total_there = list(range(0,len(fin_PDB_ids)))
    missing = [item for item in total_there  if item not in all_chk_unique]
#    print("UniGene indexes doesn't has SITE info: ",missing)
    return training_data_Uni_Gene_indexes, testing_data_Uni_Gene_indexes, missing 


def index_data(index, fin_uni_gene, gene_sat, fin_PDB_ids):
    """
    This function chooses the data by the given indexes
    """
    Uni_Genes = []
    pdb_ids_f = []
    pdbs_missed_with_gene = []
    missed_count = 0
    for i in index:
        Uni_Genes.append(fin_uni_gene[i])
        pdb_ids = []
        for k in gene_sat[i]:
            pdb_ids.append(k)
        pdb_ids_f.append(pdb_ids)
        miss_temp =[]
        for k in fin_PDB_ids[i]:
            if k not in pdb_ids:
                miss_temp.append(k)
                missed_count = missed_count + 1
        pdbs_missed_with_gene.append(miss_temp)
    return Uni_Genes, pdb_ids_f, pdbs_missed_with_gene, missed_count


#fin_uni_gene = ONGO_fin_uni_gene
#fin_PDB_ids = ONGO_fin_PDB_ids
#gene_sat = ONGO_gene_sat

#%%
def train_test_pikle(name,fin_PDB_ids, fin_uni_gene,  gene_sat, site_sat_occurance_counter,  thrs_sat_occurance_counter):
    """
   
    For training the models the number of PDBs appearing in the UniGene is selected in the 
    ascending order with the overlapping sliding window of size 4, then the first one in the 
    list is assigned for testing the rest is used for training, when the overlap ends (window size < 4)
    just chooses the whole set for training.This will make sure (75% <= training data of Uni_Gene% <= 86% 
    if the windows are more than 1)
    
    Extra info:  From the results obtained from "SITE_thrshold_info_select_training.xlsx"
    almost 80%(79.96%) of the Uni_Genes has less than 21 PDBs(that has SITE information).
    
    """
    training_data, testing_data, UniGene_index, Unique_site_sat_occurance_counter_assending_key = indexes_for_training_and_testing(site_sat_occurance_counter)
    training_data_Uni_Gene_indexes, testing_data_Uni_Gene_indexes, missing =  index_chooses_training_testing(fin_uni_gene, fin_PDB_ids, training_data,  gene_sat, testing_data, UniGene_index, Unique_site_sat_occurance_counter_assending_key)
    """
    then finalize the PDB_ids and other details for training and testing 
    and redundant: this may be used for testing(in NN-model)
                   for the MOTIF project 
                               this can be used for predicting the SITEs
                               which hasn't found yet 
    """
    
        
    training_Uni_Genes, training_pdb_ids, training_pdbs_missed_with_gene, training_missed_count = index_data(training_data_Uni_Gene_indexes, fin_uni_gene, gene_sat,fin_PDB_ids)        
    testing_Uni_Genes, testing_pdb_ids, testing_pdbs_missed_with_gene,testing_missed_count = index_data(testing_data_Uni_Gene_indexes, fin_uni_gene, gene_sat, fin_PDB_ids)
    redundant_Uni_Genes, redundant_pdb_ids,_,_ = index_data(missing, fin_uni_gene, fin_PDB_ids ,fin_PDB_ids)
    
    # training data pickle   
    pickle.dump(training_Uni_Genes, open( ''.join([name,"_training_Uni_Genes.p"]), "wb" ) ) 
    pickle.dump(training_pdb_ids, open( ''.join([name,"_training_pdb_ids.p"]), "wb" ) ) 
    pickle.dump(training_pdbs_missed_with_gene, open( ''.join([name,"_training_pdbs_missed_with_gene.p"]), "wb" ) ) 
    # testing data pickle
    pickle.dump(testing_Uni_Genes, open( ''.join([name,"_testing_Uni_Genes.p"]), "wb" ) ) 
    pickle.dump(testing_pdb_ids, open( ''.join([name,"_testing_pdb_ids.p"]), "wb" ) ) 
    pickle.dump(testing_pdbs_missed_with_gene, open( ''.join([name,"_testing_pdbs_missed_with_gene.p"]), "wb" ) ) 
    # redundant data pickle   
    pickle.dump(redundant_Uni_Genes, open( ''.join([name,"_redundant_Uni_Genes.p"]), "wb" ) ) 
    pickle.dump(redundant_pdb_ids, open( ''.join([name,"_redundant_pdb_ids.p"]), "wb" ) ) 

    print("# of PDBs in Train           : ",len(sum(training_pdb_ids, []))) 
    print("# of PDBs missing in Train(%): ",100*training_missed_count/len(sum(training_pdb_ids, []))) 
    print("# of PDBs in Test            : ",len(sum(testing_pdb_ids, [])))
    print("# of PDBs missing in Test (%): ",100*testing_missed_count/len(sum(testing_pdb_ids, []))) 