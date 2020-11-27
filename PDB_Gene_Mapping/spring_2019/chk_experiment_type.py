# -*- coding: utf-8 -*-
"""
Created on %21-Jan-2019

@author: %A.Nishanth C00294860
loading the experiment type details
"""
import pickle
import os
#%% For Tier_1
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"


"BLOCK 1 out of 7"
name = "ONGO"
def site_type_print(name):
    os.chdir("/")
    os.chdir(saving_dir)
    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    i =0
    for pdbs_info in experiment_type:
        j =0 
        for pdb_info in pdbs_info:
            for site_info in pdb_info:
                if site_info != "SOFTWARE                                              ":
#                    print(site_info)
                    print(pdb_details[i][j])
            j = j +1 
        i = i+1
#pickle.dump(gene_sat, open( ''.join([self.name,"_SITE_satisfied.p"]), "wb" ))      
site_type_print("ONGO")
print("TSG")
site_type_print("TSG")
print("Fusion")
site_type_print("Fusion")