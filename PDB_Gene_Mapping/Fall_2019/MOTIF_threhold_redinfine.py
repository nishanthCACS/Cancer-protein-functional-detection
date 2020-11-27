# -*- coding: utf-8 -*-
"""
Created on %31-12-2019

@author: %A.Nishanth C00294860

First find out the PDBs whch has SITE provided by real experiments
"""
import os
import numpy as np
import csv
from copy import deepcopy
import pickle

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
nn_probability='F:/Suceed_BIBM_1/BIBM_81_1084/rest_pikles'
os.chdir('/')
os.chdir(nn_probability)
overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
name = "ONGO"
#name = "TSG"
#name = "Fusion"

#def pdb_details_fun(name,loading_dir):
os.chdir("/")
os.chdir(loading_dir)
pdbs_not_software = pickle.load(open(''.join([name,"_pdbs_not_software.p"]), "rb" ))
pdbs_not_software_exp_type =pickle.load(open(''.join([name,"_pdbs_not_software_exp_type.p"]),"rb"))
pdbs_not_software_gene = pickle.load(open(''.join([name,"_pdbs_not_software_gene.p"]), "rb")) 
#    name='ONGO'    


'''This is for checking problem in tets set'''
loading_dir_test_gene ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/suceed_unigene"   

os.chdir("/")
os.chdir(loading_dir_test_gene)
test_pdb_ids_details  = pickle.load(open( ''.join([name,"_test_pdb_ids_details.p"]), "rb" ) )
test_unigene_details =  pickle.load(open( ''.join([name,"_test_uni_gene.p"]), "rb" )) 
for pdbs_not_software_gene_1 in pdbs_not_software_gene:
    if pdbs_not_software_gene_1 in test_unigene_details:
        for i in range(0,len(test_unigene_details)):
            if test_unigene_details[i]==pdbs_not_software_gene_1:
                print("Unigene: ",pdbs_not_software_gene_1, ' PDBs ',len(test_pdb_ids_details[i]))
                print("Unigene: i",i)
                for pdbs in test_pdb_ids_details[i]:
                    if ''.join([pdbs,'.npy']) in overlapped_PDBs_ONGO:
                        print("Provblem ;;")
                    if ''.join([pdbs,'.npy']) in overlapped_PDBs_TSG:
                        print("Provblem ;;")
                    
#%%