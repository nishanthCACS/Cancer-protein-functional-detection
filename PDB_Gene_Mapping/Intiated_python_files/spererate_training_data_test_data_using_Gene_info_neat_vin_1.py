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

 #%%
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
    # first order the site_sat_occurance_counter in assending order
    UniGene_assending_order =  [i[0] for i in sorted(enumerate(site_sat_occurance_counter), key=lambda x:x[1])]
    
    training_data =[]
    testing_data =[]
    cumulative_count = sum(site_sat_occurance_counter)
    
    m = 0 # index of UniGene
    
    carry = 0 # to hold the number of elements take for the next one   intial condition
    while cumulative_count > 0:
        i = UniGene_assending_order[m]
        if carry < 0:
            carry = 0 
        for j in range(carry,site_sat_occurance_counter[i]-3,4):
            testing_data.append(i) 
            for k in range(0,3):
                training_data.append(i)
        
        # to staisfy the ending condition
        cond_2 = True
        if cumulative_count < carry: 
            cond_2 = False
            for j in range(0,cumulative_count):
                training_data.append(i)
            break
                
        if cond_2:
            new_count = site_sat_occurance_counter[i]-carry    
            if new_count >= 4:# since 4 is the window size
                # inorder to add the remaining data for training
                # to add the front part left
                for j in range(0,carry):
                    training_data.append(i)
        
                # to add the end part left
                if cumulative_count < 4: 
                    if (new_count % 4 != 0):
                        c = list(range(site_sat_occurance_counter[i] - (new_count % 4) ,site_sat_occurance_counter[i]))
                        for k in c:
                            training_data.append(i)
                else:
                    if (new_count % 4 != 0):
                        c = list(range(site_sat_occurance_counter[i] -(new_count % 4) ,site_sat_occurance_counter[i]))
                        cond = True # to place the first element of the stack to the test data
                        carry = 4
                        for k in c:
                            if cond and (cumulative_count - site_sat_occurance_counter[i])>0:
                                testing_data.append(i)
                                cond= False
                            else:
                                training_data.append(i)
                            carry = carry-1
            else: 
                # in this case consider about overlap front and end
                # to add the front part left
                if carry > 0 and site_sat_occurance_counter[i]>0:
                    if site_sat_occurance_counter[i]>carry:
                        for j in range(0,carry):
                            training_data.append(i)
                        testing_data.append(i)
                        COUNT = 0
                        for j in range(0, site_sat_occurance_counter[i] -1 - carry):
                            training_data.append(i)
                            COUNT = COUNT +1
                            
                        carry = 3 - COUNT 
                    else:
                        for j in range(0,site_sat_occurance_counter[i]):
                            training_data.append(i)
                        carry = carry - site_sat_occurance_counter[i]
                elif  site_sat_occurance_counter[i]>0:
                    carry = 4
                    testing_data.append(i)
                    for j in range(1,site_sat_occurance_counter[i]):
                        training_data.append(i)
                        carry = carry-1
                    carry = carry - site_sat_occurance_counter[i]
        cumulative_count = cumulative_count - site_sat_occurance_counter[i]
        m =m + 1
        
    print("Actual: ", sum(site_sat_occurance_counter))
    print("GOT   : ", len(training_data)+len(testing_data))
    print("Training_percentage: ",100*(len(training_data)/sum(site_sat_occurance_counter)))
    return training_data,testing_data

training_data,testing_data = indexes_for_training_and_testing(ONGO_site_sat_occurance_counter)
#%% pre select the PDB_groups which has all site information for trainnig

#%%
def training_data_stat_claculate(thrs_sat_occurance_counter,site_sat_occurance_counter):
    """
   
    For training the models the number of PDBs appearing in the UniGene is selected in the 
    ascending order with the overlapping sliding window of size 4, then the first one in the 
    list is assigned for testing the rest is used for training, when the overlap ends (window size < 4)
    just chooses the whole set for training.This will make sure (75% <= training data of Uni_Gene% <= 86% 
    if the windows are more than 1)
    
    Extra info:  From the results obtained from "SITE_thrshold_info_select_training.xlsx"
    almost 80%(79.96%) of the Uni_Genes has less than 21 PDBs(that has SITE information).
    
    """
    missing_info_of_site_groups_more = []
#%% count the total number of occurance in the preselected gene indexes
total_sat = 0
for k in pre_selected_gene_indexes:
    total_sat = total_sat + site_sat_occurance_counter[k]
print(total_sat)   
#%%
total = sum(site_sat_occurance_counter)
# from that choose the 80% of the genes data for training 
train_size = math.ceil(len(gene_sat)*0.8)