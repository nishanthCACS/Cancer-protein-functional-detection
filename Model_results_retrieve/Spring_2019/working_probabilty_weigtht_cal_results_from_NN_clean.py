# -*- coding: utf-8 -*-
"""
Created on  %(2-Feb-2019) 10.16Am

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
#working_dir_part = "C:/Users/c00294860/Documents/Projects_UL/Continues BIBM/"
SITE_MUST = True
clean = True

Tier = 1
"""FOR clean new_ran_clean"""
working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_NN_results"])
unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean"])
saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/"])

#%% inorder to create the final formatted .csv file
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
    elif name[-5]=="l": 
        if name[-8]=='n':
            type_g = "Fusion_cl"
        elif name[-8]=='O':
            type_g = "ONGO_cl"
        elif name[-8]=='G':
            type_g = "TSG_cl" 
    # then create the .csv folder supporting format
    f= open(''.join(["T_",str(Tier),"_",type_g,'.csv']),"w+")
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
#for i in range(0,len(files)):
#    if files[i][0:3] != "T_1" and files[i][-3:-1] =="cs":
#        change_format_torch_op(files[i],Tier)
#        print(files[i], " done")
#%% Inorder to produce the results for method1..3 use this function
from class_clean_probabilty_weigtht_cal_results_from_NN_test  import clean_probability_weight_cal_results_from_NN

def finalised_results_creation(name,SITE_MUST,clean,Tier,working_dir_NN_prob ,unigene_test_dir,saving_dir_par):
    """
    This function create objects and produce the finalised results
    """
    if SITE_MUST and clean:
        working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/SITE_MUST_",name])
    elif clean:
        working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/",name])
    
    saving_dir =  "".join([saving_dir_part,name])
    if Tier == 1:
        name_cl=  "".join(["".join(["T_1_",name,".csv"])])
#        name_cl = "".join(["".join(["T_1_",name,"_cl.csv"])])
    elif Tier == 2:
#        name_csv =  "".join(["".join(["T_2_",name,".csv"])])
        name_cl = "".join(["".join(["T_2_",name,".csv"])])
#    obj1 = clean_probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir,name,name_csv,minimum_prob_decide)
#    obj1.documentation_prob()
#    del obj1 # delete the object to clear the memory
    print("Normal documentation of ", name," done")
    if clean:
        print("clean_documenation of ",name," started")
        if SITE_MUST and clean:
            working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/clean_SITE_MUST_",name])
        elif clean:
            working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/clean/",name])
        
#        saving_dir =  "".join([saving_dir_part,name])
        saving_dir = saving_dir_part
        obj = clean_probability_weight_cal_results_from_NN(working_dir_fraction, working_dir_NN_prob, unigene_test_dir, saving_dir,name,name_cl,minimum_prob_decide)
#        gene_direct_prob, fin_gene_prob, Ensamble_gene_prob = obj.documentation_prob()
        obj.documentation_prob()
        del obj # delete the object to clear the memory
        print("clean documentation of ", name," done")


finalised_results_creation("ONGO",SITE_MUST,clean,Tier,working_dir_NN_prob ,unigene_test_dir,saving_dir_part)        
finalised_results_creation("TSG",SITE_MUST,clean,Tier,working_dir_NN_prob ,unigene_test_dir,saving_dir_part)     
finalised_results_creation("Fusion",SITE_MUST,clean,Tier,working_dir_NN_prob ,unigene_test_dir,saving_dir_part)          
#%% problemetric checking
import os
import pickle
name='ONGO'
working_dir_fraction = "".join([working_dir_part, "Uniprot_Mapping/Tier_",str(Tier),"_pdb_pikles_length_gene/clean_SITE_MUST_",name])
os.chdir('/')
os.chdir(working_dir_fraction)
chk = pickle.load(open('ONGO_method_2_Hs.733536.p', "rb" ))