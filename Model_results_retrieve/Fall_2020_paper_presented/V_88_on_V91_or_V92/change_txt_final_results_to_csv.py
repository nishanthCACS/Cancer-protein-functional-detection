# -*- coding: utf-8 -*-
"""
Created on %15-June-2020 at 03.28 P.m

@author: %A.Nishanth C00294860
"""

import os

#change_files_directory='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_gene/2020_k_fold_results/Tier_1/Fully_clean/SITE/before_thresh/21_amino_acid/best_BIR_k3_to_5/'
#change_files_directory='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_gene/2020_k_fold_results/Tier_2/Fully_clean/SITE/before_thresh/21_amino_acid/best_BIR_k3_to_5/'
change_files_directory='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/clean_Finalised_probabilities_PDBs/2020_k_fold_results/best_BIR_k3_to_5/'

os.chdir('/')
os.chdir(change_files_directory)
files_there=os.listdir()
files_selected=[]
for f in files_there:
    if '.txt' in f:
        files_selected.append(f)
#%%
import shutil
for f in files_selected:
    src=''.join([change_files_directory,f])    
    dst=''.join([change_files_directory,'csv/',f[0:-4],'.csv'])  
    shutil.copy(src,dst)