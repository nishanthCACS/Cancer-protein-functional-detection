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

def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    uni_gene = pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ) )

    return pdb_details,experiment_type,uni_gene


#%%
def SITE_INFO_extractor(name,saving_dir,loading_dir):
    
    pdb_details,experiment_type,uni_gene = pdb_details_fun(name,loading_dir)

    pdbs_not_software=[]
    pdbs_not_software_exp_type=[]
    pdbs_not_software_gene=[]
    pdbs_not_software_must=[]
    pdbs_not_software_must_exp_type=[]
    pdbs_not_software_must_exp_type_gene=[]
    chk_one=False
    for i in range(0,len(experiment_type)):
        for j in range(0,len(experiment_type[i])):
            software_method=False
            other_type = False
            type_exp = experiment_type[i][j]
            for t_exp in type_exp:
                if t_exp.split()[0]=='SOFTWARE':
                    software_method=True
                else:
                    other_type=True
            if not software_method:
                chk_one=True
                pdbs_not_software_must.append(pdb_details[i][j])
                pdbs_not_software_must_exp_type.append(experiment_type[i][j])
                pdbs_not_software_must_exp_type_gene.append(uni_gene[i])
            if other_type:
                pdbs_not_software.append(pdb_details[i][j])
                pdbs_not_software_exp_type.append(experiment_type[i][j])
                pdbs_not_software_gene.append(uni_gene[i])
    '''Since the number of satisfied PDBs are low just take all the Author contain PDBs'''
    os.chdir('/')
    os.chdir(saving_dir)  
    pickle.dump(pdbs_not_software, open(''.join([name,"_pdbs_not_software.p"]), "wb"))  
    pickle.dump(pdbs_not_software_exp_type, open(''.join([name,"_pdbs_not_software_exp_type.p"]), "wb"))  
    pickle.dump(pdbs_not_software_gene, open(''.join([name,"_pdbs_not_software_gene.p"]), "wb"))  
    if chk_one:
        pickle.dump(pdbs_not_software_must, open(''.join([name,"_pdbs_not_software_must.p"]), "wb"))  
        pickle.dump(pdbs_not_software_must_exp_type, open(''.join([name,"_pdbs_not_software_must_exp_type.p"]), "wb"))  
        pickle.dump(pdbs_not_software_must_exp_type_gene, open(''.join([name,"_pdbs_not_software_must_gene.p"]), "wb"))  
        print(name, " SITE info saved")
#%%
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

name = "ONGO"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
SITE_INFO_extractor(name,saving_dir,loading_dir)
name = "TSG"
SITE_INFO_extractor(name,saving_dir,loading_dir)
name = "Fusion"
SITE_INFO_extractor(name,saving_dir,loading_dir)
#%%