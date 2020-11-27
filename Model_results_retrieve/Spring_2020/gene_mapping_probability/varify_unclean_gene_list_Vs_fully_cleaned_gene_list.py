# -*- coding: utf-8 -*-
"""
Created on %(02-02-2020) 11.54Am

@author: %A.Nishanth C00294860
"""
import os
import pickle
#%%
def checking_the_pdb_gene_lists(name, clean_must_sat_dir,threshold_dir,num_aminoacids):
    working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
    unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_1/clean"])
    os.chdir('/')
    os.chdir(unigene_test_dir)
    test_gene_list = pickle.load(open( "".join([name,"_test_uni_gene.p"]),"rb"))
    train_clean_gene_list = pickle.load(open("".join([name,"_train_unigene_details.p"]),"rb"))
    train_old_genes=train_clean_gene_list+test_gene_list
    #%
    working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
    working_dir_fraction = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
    os.chdir('/')
    os.chdir(working_dir_fraction)
    items=os.listdir()
    sat_genes=[]
    for ids in items:
       chk_str=ids.split("_")
       if chk_str[1]=='direct':
           sat_genes.append(chk_str[2][0:-2])
       elif  chk_str[1]=='method':
           if chk_str[2]=='1':      
               sat_genes.append(chk_str[3][0:-2])
    may_prob=[]
    for ids in train_old_genes:
       if ids not in sat_genes:
           print(ids," not in new 21 amino acid")
           may_prob.append(ids)
    #% check whether atleat one PDBs satisfied these genes
    #first retrive all the previously encountered PDBs
    os.chdir('/')
    os.chdir(working_dir)
    Ids_info_all = pickle.load(open( ''.join([name,"_Ids_info_all.p"]), "rb" ) )
    uni_gene = pickle.load(open( ''.join([name,"_uni_gene.p"]), "rb" ) )
    prob_genes_info_all=[]
    for ids in may_prob:
        for i in range(0,len(uni_gene)):
            if uni_gene[i]==ids:
                prob_genes_info_all.append(Ids_info_all[i])
    #% then gothrough the details and check wther it satisfied the length condition
    threshold_length=81
    os.chdir('/')
    os.chdir(clean_must_sat_dir)
    if name=='ONGO':
        site_sat_flatten=  pickle.load(open( ''.join(['OG',"_train_PDBs.p"]), "rb"))
    else:
        site_sat_flatten=  pickle.load(open( ''.join([name,"_train_PDBs.p"]), "rb"))
    
    PDBs_sel=[]
    for ids in prob_genes_info_all:
        PDB_t =[]
        for id_t in ids:
            if len(id_t) == 3:
                for j in range(0,len(id_t[2])):
                    if id_t[2][j]=="-":
                       if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > threshold_length:
                           if id_t[0] in site_sat_flatten:
                               PDB_t.append(id_t[0])    
    if len(PDBs_sel)>0:
        print("Problem in ",name)
    else:
        print("OK")
    
    test_new_gene_list=[]
    for ids in test_gene_list:
        if ids not in may_prob:
            test_new_gene_list.append(ids)
    train_new_gene_list=[]
    for ids in train_clean_gene_list:
        if ids not in may_prob:
            train_new_gene_list.append(ids)
    unigene_save_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/',str(num_aminoacids),'_amino_acids_fully_clean_SITE'])
    
    os.chdir('/')
    os.chdir(unigene_save_dir)
    pickle.dump(train_new_gene_list, open(''.join([name,"_train_genes_",str(num_aminoacids),".p"]), "wb")) 
    pickle.dump(test_new_gene_list, open(''.join([name,"_test_genes_",str(num_aminoacids),".p"]), "wb")) 

clean_must_sat_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Before_thresh/20_amino_acids/17-prop"
#clean_must_sat_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Before_thresh/21_amino_acids/21-prop"
threshold_dir= "old_thresh/20_amino_acids/Clean_no_overlap_at_all_20_amino_prop_with_SITE/"

checking_the_pdb_gene_lists("ONGO",clean_must_sat_dir,threshold_dir,20)
checking_the_pdb_gene_lists("TSG",clean_must_sat_dir,threshold_dir,20)
checking_the_pdb_gene_lists("Fusion",clean_must_sat_dir,threshold_dir,20)