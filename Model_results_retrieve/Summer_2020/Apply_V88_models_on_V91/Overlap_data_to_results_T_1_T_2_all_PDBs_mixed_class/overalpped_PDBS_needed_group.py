# -*- coding: utf-8 -*-
"""
Created on %31-05-2020 3.35 p.m

@author: %A.Nishanth C00294860
"""
import os
import pickle
from shutil import copy2

#%%
os.chdir('/')
os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid')
overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
os.chdir('/')
os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/Train/ONGO')
src_t_1='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/Train/ONGO/'
src_t_2='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/Train/TSG/'

dst_t='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/21_amino_acid/Train/overlapped/'
#
#os.chdir('/')
#os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid')
#overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
#overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
#os.chdir('/')
#os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid/Train/ONGO')
#src_t_1='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid/Train/ONGO/'
#src_t_2='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid/Train/TSG/'
#
#dst_t='G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/before_thresh/20_amino_acid/Train/overlapped/'
#%%
for pdb in overlapped_PDBs_ONGO:
    src=''.join([src_t_1,pdb])
    dst=''.join([dst_t,pdb])
    copy2(src, dst)
    
for pdb in overlapped_PDBs_TSG:
    src=''.join([src_t_2,pdb])
    dst=''.join([dst_t,pdb])
    copy2(src, dst)
#%% # to make sure go through again
'''
The check found there is no overlapping in Test set
'''
#os.chdir('/')
#os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/all_c_alpha_SITE/21_amino_acid/Train/ONGO')
#pdbs_ONGO=os.listdir()
#
#
#os.chdir('/')
#os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/all_c_alpha_SITE/21_amino_acid/Train/TSG')
#pdbs_TSG=os.listdir()
#
#os.chdir('/')
#os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/all_c_alpha_SITE/21_amino_acid/Train/Fusion')
#pdbs_Fusion=os.listdir()
##
##os.chdir('/')
##os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/all_c_alpha_SITE/21_amino_acid/Test/ONGO')
##pdbs_ONGO=os.listdir()
##
##
##os.chdir('/')
##os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/all_c_alpha_SITE/21_amino_acid/Test/TSG')
##pdbs_TSG=os.listdir()
##
##os.chdir('/')
##os.chdir('G:/ULL_Project/optimally_tilted/quater_models/Fully_clean/SITE/all_c_alpha_SITE/21_amino_acid/Test/Fusion')
##pdbs_Fusion=os.listdir()
#def overlap_finder(list_1,list_2_1,list_2_2):
#    list_2=list_2_1+list_2_2
#    overlapped=[]
#    for pdb in list_1:
#        if pdb in list_2:
#            overlapped.append(pdb)
#        
#    return overlapped
#
#overlap_pdbs_ONGO=overlap_finder(pdbs_ONGO,pdbs_TSG,pdbs_Fusion)
#overlap_pdbs_TSG=overlap_finder(pdbs_TSG,pdbs_ONGO,pdbs_Fusion)