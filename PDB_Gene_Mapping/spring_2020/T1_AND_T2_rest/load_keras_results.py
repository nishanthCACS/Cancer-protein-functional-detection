# -*- coding: utf-8 -*-
"""
Created on %10-Mar-2020(04.48 P.m)

@author: %A.Nishanth C00294860
"""

import os
import pickle
import numpy as np
from copy import deepcopy 
os.chdir('/')
#os.chdir('E:/BIBM_project_data/Before_threshold_changed/21_SITE/Tier_1_results')
os.chdir('F:/BIBM_back_up_13_03_2020/Tier_1_results')
files=os.listdir()
#%%
PDB_prb_dic_all=[]
for m in range(0,len(files)):
    PDB_prob_dir=''.join(['F:/BIBM_back_up_13_03_2020/Tier_1_results/',files[m]])
    os.chdir('/')
    os.chdir(PDB_prob_dir)
    i=0
    pdb_prob_dic={}
    pdbs_test_0 = pickle.load( open( ''.join(["pdbs_test_",str(i),".p"]), "rb" ))
    test_probabilities_0= pickle.load( open( ''.join(["test_probabilities_",str(i),".p"]), "rb" ))
    for j in range(0,len(pdbs_test_0)):
        for k in range(0,len(pdbs_test_0[j])):
            pdb_prob_dic[deepcopy(pdbs_test_0[j][k][0:-4])]=deepcopy(test_probabilities_0[j][k])
    
    for i in range(1,10):
        pdbs_test = pickle.load( open( ''.join(["pdbs_test_",str(i),".p"]), "rb" ))
        test_probabilities= pickle.load( open( ''.join(["test_probabilities_",str(i),".p"]), "rb" ))
        for j in range(0,len(pdbs_test)):
            for k in range(0,len(pdbs_test[j])):
                new_prob = np.add(pdb_prob_dic[pdbs_test[j][k][0:-4]],test_probabilities[j][k])
                pdb_prob_dic.update({pdbs_test[j][k][0:-4]:deepcopy(new_prob)})
    ##% then average them to 
    for pdb in pdb_prob_dic: 
        new_prob = np.round(pdb_prob_dic[pdb]/10,decimals=2)
        pdb_prob_dic[pdb] = deepcopy(new_prob)
    print(files[m],': ',pdb_prob_dic['2HYY'])
#    print(files[m],': ',pdb_prob_dic['2HZ0'])
#    print(files[m],': ',pdb_prob_dic['2HZ4'])
#    print(files[m],': ',pdb_prob_dic['2HZI'])

    PDB_prb_dic_all.append([files[m],deepcopy(pdb_prob_dic)])
#pickle.dump(pdb_prob_dic, open("pdb_prob_dic_2019_T1.p", "wb"))  

