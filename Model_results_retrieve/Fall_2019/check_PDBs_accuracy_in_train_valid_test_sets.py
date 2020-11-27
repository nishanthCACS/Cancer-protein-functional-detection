# -*- coding: utf-8 -*-
"""
Created on %30-Dec-2019 at 2.34pm

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import csv
from copy import deepcopy
import pickle
#%%
loading_dir='F:/Suceed_BIBM_1/BIBM_81_1083/extract_results'
os.chdir('/')
os.chdir(loading_dir)
OG_test_probs_numpy = np.load('OG_test_probs.npy')
TSG_test_probs_numpy = np.load('TSG_test_probs.npy')
Fusion_test_probs_numpy = np.load('Fusion_test_probs.npy')
Fusion_test_PDBs = pickle.load(open("Fusion_test_PDBs.p","rb")) 

#%%
OG_valid_probs_numpy = np.load('OG_valid_probs.npy')
TSG_valid_probs_numpy = np.load('TSG_valid_probs.npy')
Fusion_valid_probs_numpy = np.load('Fusion_valid_probs.npy')
OG_train_probs_numpy = np.load('OG_train_probs.npy')
TSG_train_probs_numpy = np.load('TSG_train_probs.npy')
Fusion_train_probs_numpy = np.load('Fusion_train_probs.npy')