# -*- coding: utf-8 -*-
"""
Created on %10-Mar-2020(05.38 P.m)

@author: %A.Nishanth C00294860
"""
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Paper_writing/Comparison_paper_results/Mapping_uniprot_mine')
ONGO_mapped_uni_gene = pickle.load(open("ONGO_mapped_uni_gene.p", "rb"))   

ONGO_unique_gene_name_given=pickle.load(open("ONGO_unique_gene_name_given.p", "rb"))   
'''
To get the unigene tyo gene actual gene name

'''
#%%
os.chdir('/')
os.chdir('E:/BIBM_project_data/Before_threshold_changed/21_SITE/Tier_1_results')
test_probabilities = pickle.load(open("pdb_prob_dic_2019_T1.p"]), "rb" )
