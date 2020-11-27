# -*- coding: utf-8 -*-
"""
Created on 24-May-2020 at 1.23p.m

@author: %A.Nishanth C00294860

For the timebeing only checked in V84 mapped PDBs and V88 newly included and mapped TSG PDBs only
All_c_alpha is the best choice to check
"""

import os
import pickle

#%%
for Train_test in ['Train','Test']:
    for name in ['ONGO','TSG','Fusion']:
        directory_21=''.join(['F:/scrach_cacs_1083/all_c_alpha_SITE/21_amino_acid/',Train_test,'/',name])
        directory_20=''.join(['F:/scrach_cacs_1083/all_c_alpha_SITE/20_amino_acid/',Train_test,'/',name])
        os.chdir('/')
        os.chdir(directory_21)
        files_21=os.listdir()
        os.chdir('/')
        os.chdir(directory_20)
        files_20=os.listdir()
        
        for pdb in files_21:
            if pdb not in files_20:
                print(Train_test,': ',name)
                print(pdb[0:-4])
