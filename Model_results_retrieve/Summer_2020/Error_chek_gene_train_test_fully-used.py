# -*- coding: utf-8 -*-
"""
Created on %23-May-2020 at 6.42pm

@author: %A.Nishanth C00294860
"""
import os
import pickle

name='Fusion'
#name='TSG'
#name='ONGO'

All_load_dir ='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene/'

os.chdir('/')
os.chdir(All_load_dir)
SITE_satisfied_PDBs= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb")) 
SITE_satisfied_gene_list_all= pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb"))
count=0
for i in range(0,len(SITE_satisfied_gene_list_all)):
    if len(SITE_satisfied_PDBs)>0:
        count=count+1
        
SITE_load_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/weights_assign_methods/used_Fully_clean_models/T_',str(1),'/SITE/before_thresh/',str(21),'_aminoacid'])
os.chdir('/')
os.chdir(SITE_load_dir)
satisfied_PDBs= pickle.load(open(''.join([name,"_satisfied_PDBs.p"]), "rb")) 
gene_list_all= pickle.load(open(''.join([name,"_satisfied_Gene_ID.p"]), "rb"))

unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'
os.chdir('/')
os.chdir(unigene_load_dir)

train_genes= pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
test_genes= pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 

genes_all=train_genes+test_genes
genes_all=list(set(genes_all))
for genes in genes_all:
    if genes not in gene_list_all:
        print(genes)
        
#%%