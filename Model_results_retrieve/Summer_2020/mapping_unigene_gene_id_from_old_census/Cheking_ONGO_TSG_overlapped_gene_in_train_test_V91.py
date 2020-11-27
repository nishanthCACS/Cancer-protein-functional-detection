# -*- coding: utf-8 -*-
"""
Created on %27-May-2020 at 21.33pm

@author: %A.Nishanth C00294860



"""
import os
import pickle
#%
'''
To make sure none of the gene_ids of test not fall into problemetric genes+

'''
unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'
os.chdir('/')
os.chdir(unigene_load_dir)
problemteric_TSG_genes = pickle.load(open('problemteric_TSG_genes.p', "rb"))  
problemteric_ONGO_genes= pickle.load(open('problemteric_ONGO_genes.p', "rb"))  
problemteric_genes_t=problemteric_TSG_genes+problemteric_ONGO_genes
problemteric_genes=[]
for i in range(0,len(problemteric_genes_t)):
    problemteric_genes.append(problemteric_genes_t[i][2])
    
TSG_test_gene_id= pickle.load(open('TSG_test_gene_id.p', "rb"))  
ONGO_test_gene_id= pickle.load(open('ONGO_test_gene_id.p', "rb"))  
Fusion_test_gene_id= pickle.load(open('Fusion_test_gene_id.p', "rb"))  
test_genes=ONGO_test_gene_id+TSG_test_gene_id+Fusion_test_gene_id
for genes in test_genes:
    if genes in problemteric_genes:
        print(genes)
        
TSG_train_gene_id= pickle.load(open('TSG_train_gene_id.p', "rb"))  
ONGO_train_gene_id= pickle.load(open('ONGO_train_gene_id.p', "rb"))  
Fusion_train_gene_id= pickle.load(open('Fusion_train_gene_id.p', "rb"))  
train_genes=ONGO_train_gene_id+TSG_train_gene_id+Fusion_train_gene_id
for genes in train_genes:
    if genes in problemteric_genes:
        print(genes)
#%%
unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
os.chdir('/')
os.chdir(unigene_load_dir)
TSG_test_gene_21= pickle.load(open('TSG_test_genes_21.p', "rb"))  
ONGO_test_gene_21= pickle.load(open('ONGO_test_genes_21.p', "rb"))  
Fusion_test_gene_21= pickle.load(open('Fusion_test_genes_21.p', "rb"))  
test_genes=ONGO_test_gene_21+TSG_test_gene_21+Fusion_test_gene_21
for genes in test_genes:
    if genes in problemteric_genes:
        print(genes)
        
TSG_train_gene_21= pickle.load(open('TSG_train_genes_21.p', "rb"))  
ONGO_train_gene_21= pickle.load(open('ONGO_train_genes_21.p', "rb"))  
Fusion_train_gene_21= pickle.load(open('Fusion_train_genes_21.p', "rb"))  
train_genes=ONGO_train_gene_21+TSG_train_gene_21+Fusion_train_gene_21
for genes in train_genes:
    if genes in problemteric_genes:
        print(genes)