# -*- coding: utf-8 -*-
"""
Created on  %(15-Nov-2018) 21.23pm

@author: %A.Nishanth C00294860
"""


import os
import csv
# make sure the 
from class_clean_probabilty_weigtht_cal_results_from_NN_test  import clean_probability_weight_cal_results_from_NN
#%%
minimum_prob_decide = 0.25 
#% First change the results given by Torch to only the pdb_id formating
working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"

SITE_MUST = True

"""For intialising the directorie for code"""
working_dir_NN_prob = "".join([working_dir_part, "a_Results_after_success/clean_NN_results"])
unigene_test_dir = "".join([working_dir_part,"a_Results_after_success/clean"])
saving_dir_part = "".join([working_dir_part, "a_Results_after_success/clean_Finalised_probabilities_gene"])

#%% inorder to create the final formatted .csv file
def change_format_torch_op(name):
    """
    This function change the given files format to the class 
    class_probabilty_weigtht_cal_results_from_NN  
    
    readable format that the first coloumn should be in the format of pdb_id
    """
    f = open(name)
    csv_f = csv.reader(f)
    
    stack=[]
    for row in csv_f:
        stack.append(row)
    r_stack=[]    
    for ele in stack:
        pdb_name = ele[0][-8:-4]
#        print(pdb_name)
        temp = [pdb_name]
        for i in range(1,len(ele)):
            temp.append(ele[i])
        r_stack.append(temp)
    # then create the .csv folder supporting format
    f= open(''.join([name[0:-4],'_processed.csv']),"w+")
    for rw in r_stack:
        #insert the group details
        for s in range(0,len(rw)-1):
            f.write(rw[s]+',')
        f.write(rw[-1]+'\n')  
    f.close() 
#%% produce the initial results
os.chdir('/')
os.chdir(working_dir_NN_prob)
files = os.listdir() 
name_NN = files[0]
print(name_NN, " intiated")
change_format_torch_op(name_NN)
print(name_NN, " done")
#%% Inorder to produce the results for method1..3 use this function
from class_clean_probabilty_weigtht_cal_results_from_NN_test  import clean_probability_weight_cal_results_from_NN

def finalised_results_creation(name,working_dir_NN_prob ,unigene_test_dir,saving_dir_par):
    """
    This function create objects and produce the finalises results
    """
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/pdb_pikles_length_gene/",name])
    name_csv =  "".join(["".join([name_NN[0:-4],'_processed.csv'])])

    obj = clean_probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir_part,name,name_csv,minimum_prob_decide)
    obj.documentation_prob()
    del obj # delete the object to clear the memory
    print("Documentation ", name," done")
    
#%% to retrieve the names
naming_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
os.chdir('/')
os.chdir(naming_dir)
files=os.listdir()
for names in files:
    name = names[8:-4]
    print("Going to work on: ",name)
    finalised_results_creation(name,working_dir_NN_prob ,unigene_test_dir,saving_dir_part)        

#%% then do it for the cleaned data

"""
Error handling old
"""
import pickle
os.chdir('/')
os.chdir(saving_dir_part)
gene_direct_prob = pickle.load( open(''.join([name,"_gene_direct_prob.p"]), "rb" ))
fin_gene_prob = pickle.load( open(''.join([name,"_fin_gene_prob.p"]), "rb" ))
Ensamble_gene_prob = pickle.load( open(''.join([name,"_Ensamble_gene_prob.p"]), "rb" ))
#%%
weight_initialize_gene_from_PDB_seq_length = ["Fusion_T_2_direct_Hs.496666.p","Fusion_T_2_direct_Hs.94070.p","Fusion_T_2_method_1_Hs.112408.p","Fusion_T_2_method_1_Hs.116237.p"]
for n in weight_initialize_gene_from_PDB_seq_length:
            print(n)
#            print( n[len(name)+1:len(name)+7])
            if  n[len(name)+1:len(name)+7]=='direct':
                print(n[len(name)+8:-2])   
            elif n[len(name)+1:len(name)+9]=='method_1':
                print(n[len(name)+10:-2])   

#import os
#import csv
#import pickle
#minimum_prob_decide = 0.25
#name = "ONGO"
#"""FOR clean new_ran_clean"""
#working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_NN_results"])
#unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean"])
#saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/"])
#
#if Tier == 1:
#    name_csv =  "".join(["".join(["T_1_",name,".csv"])])
#    name_cl = "".join(["".join(["T_1_",name,"_cl.csv"])])
#elif Tier == 2:
#    name_csv =  "".join(["".join(["T_2_",name,".csv"])])
#    name_cl = "".join(["".join(["T_2_",name,"_cl.csv"])])
#
#if SITE_MUST and clean:
#        working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/SITE_MUST_",name])
#elif clean:
#        working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/",name])
#if clean:
#    print("clean_documenation of ",name," started")
#    if SITE_MUST and clean:
#        working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/clean_SITE_MUST_",name])
#    elif clean:
#        working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/clean_",name])
#
#Tier = 1
#
#"""
#Here place the problemetric pikle
#"""
#name = "ONGO"
#
#problemetic_pikle ='ONGO_method_1_Hs.165950.p'
#
#
#os.chdir('/')
#os.chdir(working_dir_NN_prob)
#
## make sure which Tier is working on
#f = open(name_cl)
#csv_f = csv.reader(f)
#
#PDB_id = []
#probabilities =[]
#cond = True
#for row in csv_f:
#    if cond:
#        #to avoid the heading
#        cond = False
#    else:
#        PDB_id.append(row[0])
#        #order is OG, TSG, Fusion
#        probabilities.append([float(row[1]),float(row[2]),float(row[3])])
#
#os.chdir('/')
#os.chdir(working_dir_fraction)
#
#whole = pickle.load(open(problemetic_pikle,"rb"))
## go through all available coverages
## and calculate the overall fraction first for all PDBs
#pdb_temp = []
#sum_length = 0
#for j in range(0,len(whole[1])):
#    sum_length = sum_length + whole[1][j]
#    for k in range(0,len(whole[3])):
#        #intially needed to append the fractions 
#        if j==0:
#            pdb_temp.append(whole[1][j]*whole[2][j][k])
#        else:
#            pdb_temp[k] = pdb_temp[k] + whole[1][j]*whole[2][j][k]
# 
#for k in range(0,len(whole[3])):            
#    pdb_temp[k] = pdb_temp[k]/sum_length
#    
## then gothrough probabilities and caculate the final
#p_OG = 0
#p_TSG = 0
#p_Fusion = 0
#for k in range(0,len(whole[3])): 
#    if pdb_temp[k] > 0:
#        for j in range(0,len(PDB_id)):
#            if whole[3][k] == PDB_id[j]:
#                p_OG = p_OG + pdb_temp[k]*probabilities[j][0]
#                p_TSG = p_TSG + pdb_temp[k]*probabilities[j][1]
#                p_Fusion = p_Fusion + pdb_temp[k]*probabilities[j][2]                