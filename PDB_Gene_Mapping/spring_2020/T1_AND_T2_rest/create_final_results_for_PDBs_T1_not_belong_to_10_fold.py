# -*- coding: utf-8 -*-
"""
Created on %10-Mar-2020(05.44 P.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle
from copy import deepcopy

def retrieve_the_results(Tier,test_probabilities):
    retrievable_PDBs=[] 
    for pdb in test_probabilities:
        retrievable_PDBs.append(pdb)
    '''
    use the retrievable PDBs onlyh to craete the results for Genes finalised Probability
    go through the T_1 and check the where these PDBs classes belong
    To select the PDBs of Tier 1
    '''
    Tier=1
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked')
    files=os.listdir()
    SITE_satisfied_files=[]
    SITE_satisfied_PDBs_apart=[]
    gene_list=[]
    
    for i in range(0,len(files)):
        if files[i][-16:-2]=='SITE_satisfied':      
            if files[i][0:-17][-1]==str(Tier):
                SITE_satisfied_files.append(files[i])
                gene_load_name=''.join([files[i][0:-17],'_thres_sat_gene_list.p'])
                temp_load_gene = pickle.load(open(gene_load_name, "rb" ))
                temp_load_PDBs = pickle.load(open(files[i], "rb" ))
                final_PDBs=[]
                final_genes=[]
                for m in range(0,len(temp_load_PDBs)):
                    if len(temp_load_PDBs[m])>0:
                        pdb_sel=[]
                        for pdb in temp_load_PDBs[m]:
                            if pdb in retrievable_PDBs:
                                pdb_sel.append(pdb)
                        if len(pdb_sel)>0:
                            final_PDBs.append(pdb_sel)
                            final_genes.append(temp_load_gene[m])                  
                SITE_satisfied_PDBs_apart.append(deepcopy(final_PDBs))
                gene_list.append(deepcopy(final_genes))
    return gene_list,SITE_satisfied_PDBs_apart,SITE_satisfied_files
#%%
os.chdir('/')
os.chdir('E:/BIBM_project_data/Before_threshold_changed/21_SITE/Tier_1_results')
test_probabilities = pickle.load(open("pdb_prob_dic_2019_T1.p", "rb"))   
#gene_list,SITE_satisfied_PDBs_apart,SITE_satisfied_files =retrieve_the_results(1,test_probabilities)
#
#test_probabilities = pickle.load(open("ONGO_mapped_uni_gene.p", "rb"))   