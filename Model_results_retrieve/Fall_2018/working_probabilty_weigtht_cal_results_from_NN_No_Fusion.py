# -*- coding: utf-8 -*-
"""
Created on %(23-Oct-2018) 18.27Pm

@author: %A.Nishanth C00294860
"""
import os
import csv
import pickle
# make sure the 
from class_probabilty_weigtht_cal_results_from_NN_No_Fusion_test  import probability_weight_cal_results_from_NN_NOF
#%%
minimum_prob_decide = 0.25
#SITE_MUST = True
SITE_MUST = False
print("Make sure SITE_Must case")
if SITE_MUST:
    print("***********-----************")
    print("")
    print("You took SITE_MUST case")
    print("")
else:
    print("***********-----************")
    print("")
    print("You took whole case")
    print("")
    
    
#%% First change the reuslts given by Torch to only the pdb_id formating
working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#working_dir_part = "C:/Users/c00294860/Documents/Projects_UL/Continues BIBM/"
if SITE_MUST:
    """FOR new_ran_SITE_MUST"""
    working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_SITE_MUST_NN_results"])
    unigene_test_dir = "".join([working_dir_part,'a_Results_for_model_train/new_ran_SITE_must'])
    saving_dir_part =  "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_SITE_MUST_Finalised_probabilities_gene/"])
else:
    """FOR new_ran_whole"""
    working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_whole_NN_results"])
    unigene_test_dir = "".join([working_dir_part,'a_Results_for_model_train/Tier_1/new_random_whole'])
    saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_whole_Finalised_probabilities_gene/"])

Tier = 1
#%%
def change_format_torch_op(name,Tier):
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
    if name[-5]=='n':
        type_g = "Fusion"
    elif name[-5]=='O':
        type_g = "ONGO"
    elif name[-5]=='G':
        type_g = "TSG"    
    # then create the .csv folder supporting format
    f= open(''.join(["T_",str(Tier),"_",type_g,'.csv']),"w+")
    for rw in r_stack:
        #insert the group details
        for s in range(0,len(rw)-1):
            f.write(rw[s]+',')
        f.write(rw[-1]+'\n')  
    f.close() 
#%%
os.chdir('/')
os.chdir(working_dir_NN_prob)
files = os.listdir() 
#%%
for i in range(0,len(files)):
    if files[i][0:3] != "T_1" and files[i][-3:-1] =="cs":
        change_format_torch_op(files[i],Tier)
        print(files[i], " done")
    #%%
"BLOCK 1 out of 7"
name = "ONGO"
if SITE_MUST:
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/SITE_MUST_",name])
else:
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([saving_dir_part,name])
#os.mkdir(saving_dir)
ONGO = probability_weight_cal_results_from_NN_NOF(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir,name,minimum_prob_decide)
ONGO.documentation_prob()
"BLOCK 2 out of 7"
name = "TSG"
if SITE_MUST:
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/SITE_MUST_",name])
else:
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([saving_dir_part,name])
#os.mkdir(saving_dir)
TSG = probability_weight_cal_results_from_NN_NOF(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir,name,minimum_prob_decide)
TSG.documentation_prob()

"BLOCK 3 out of 7"
name = "Fusion"
if SITE_MUST:
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/SITE_MUST_",name])
else:
    working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([saving_dir_part,name])
#os.mkdir(saving_dir)
Fusion = probability_weight_cal_results_from_NN_NOF(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir,name,minimum_prob_decide)
Fusion.documentation_prob()
#%% Unwanted checked earlier
##% read the Gene_ids from the working_dir_fraction 
#name = "Fusion"
#working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/",name])    
#os.chdir('/')
#os.chdir(working_dir_fraction)
#files = os.listdir() 
#
#direct_pikles_UniGene = []
#method_pikles_UniGene = []
#
#direct_pikles = []
#method_1_pikles = []
#method_2_pikles = []
#method_3_pikles = []
#for n in files: 
#    if n[len(name)+1:len(name)+7] == 'direct':
#        direct_pikles.append(n)
#        if name == "ONGO":
#           direct_pikles_UniGene.append(n[12:-2])
#        elif name == "TSG":
#            direct_pikles_UniGene.append(n[11:-2])
#        elif name == "Fusion":
#            direct_pikles_UniGene.append(n[14:-2])
#    elif n[len(name)+1:len(name)+9] == 'method_1':
#        method_1_pikles.append(n)
#    elif n[len(name)+1:len(name)+9] == 'method_2':
#        method_2_pikles.append(n)
#    elif n[len(name)+1:len(name)+9] == 'method_3':
#        method_3_pikles.append(n)
#        if name == "ONGO":
#            method_pikles_UniGene.append(n[14:-2])
#        elif name == "TSG":
#            method_pikles_UniGene.append(n[13:-2])
#        elif name == "Fusion":
#            method_pikles_UniGene.append(n[16:-2])
#            
##% then check the taken UniGene in the given Test UniGene list
## loading the  
#unigene_test_dir = "".join([working_dir_part,'a_Results_for_model_train/Tier_1/new_random_whole'])
##os.chdir('/')
##os.chdir(unigene_test_dir)
##test_pickle = "".join([name,'_test_uni_gene.p'])
##test_gene_list = pickle.load(open(test_pickle,"rb"))
##"""
##here you check the direct_pikles_UniGene or method_pikles_UniGene in the UniGene in test_gene_list and do further
##"""
##wanted_direct_pikles = []
##wanted_method_1_pikles = []
##wanted_method_2_pikles = []
##wanted_method_3_pikles = []
##for i in range(0,len(method_pikles_UniGene)):
##    if method_pikles_UniGene[i] in test_gene_list:
##        wanted_method_1_pikles.append(method_1_pikles[i])
##        wanted_method_2_pikles.append(method_2_pikles[i])
##        wanted_method_3_pikles.append(method_3_pikles[i])
##for i in range(0,len(direct_pikles_UniGene)):
##    if direct_pikles_UniGene[i] in test_gene_list:        
##        wanted_direct_pikles.append(direct_pikles[i])
##% For Tier_1
"BLOCK 1 out of 7"
name = "ONGO"
working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([saving_dir_part,name])
#os.mkdir(saving_dir)
ONGO = probability_weight_cal_results_from_NN_NOF(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir,name,minimum_prob_decide)
ONGO.documentation_prob()
#%% 
"""
Error handling
"""
import os
import csv
import pickle
minimum_prob_decide = 0.25

#% First change the reuslts given by Torch to only the pdb_id formating
working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#working_dir_part = "C:/Users/c00294860/Documents/Projects_UL/Continues BIBM/"
"""FOR new_ran_whole"""
#working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_whole_NN_results"])
#unigene_test_dir = "".join([working_dir_part,'a_Results_for_model_train/Tier_1/new_random_whole'])
#saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_whole_Finalised_probabilities_gene/"])
"""FOR new_ran_SITE_MUST"""
working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_SITE_MUST_NN_results"])
unigene_test_dir = "".join([working_dir_part,'a_Results_for_model_train/new_ran_SITE_must'])
saving_dir_part =  "".join([working_dir_part, "a_Results_for_model_train/Tier_1/new_random_SITE_MUST_Finalised_probabilities_gene/"])

Tier = 1

"""
Here place the problemetric pikle
"""
name = "ONGO"

problemetic_pikle ='ONGO_method_2_Hs.370770.p'#'ONGO_method_2_Hs.132966.p'#

working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([saving_dir_part,name])

os.chdir('/')
os.chdir(working_dir_NN_prob)

# make sure which Tier is working on
f = open("".join(["T_1_",name,".csv"]))
csv_f = csv.reader(f)

PDB_id = []
probabilities =[]
cond = True
for row in csv_f:
    if cond:
        #to avoid the heading
        cond = False
    else:
        PDB_id.append(row[0])
        #order is OG, TSG, Fusion
        probabilities.append([float(row[1]),float(row[2]),float(row[3])])

os.chdir('/')
os.chdir(working_dir_fraction)

whole = pickle.load(open(problemetic_pikle,"rb"))
# go through all available coverages
# and calculate the overall fraction first for all PDBs
pdb_temp = []
sum_length = 0
for j in range(0,len(whole[1])):
    sum_length = sum_length + whole[1][j]
    for k in range(0,len(whole[3])):
        #intially needed to append the fractions 
        if j==0:
            pdb_temp.append(whole[1][j]*whole[2][j][k])
        else:
            pdb_temp[k] = pdb_temp[k] + whole[1][j]*whole[2][j][k]
print("sum_length: ",sum_length)
for k in range(0,len(whole[3])):            
    pdb_temp[k] = pdb_temp[k]/sum_length
    
# then gothrough probabilities and caculate the final
p_OG = 0
p_TSG = 0
p_Fusion = 0
for k in range(0,len(whole[3])): 
    if pdb_temp[k] > 0:
        for j in range(0,len(PDB_id)):
            if whole[3][k] == PDB_id[j]:
                p_OG = p_OG + pdb_temp[k]*probabilities[j][0]
                p_TSG = p_TSG + pdb_temp[k]*probabilities[j][1]
                p_Fusion = p_Fusion + pdb_temp[k]*probabilities[j][2]

#if p_OG == 0 and  p_TSG == 0 and  p_Fusion == 0:              