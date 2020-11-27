# -*- coding: utf-8 -*-
"""
Created on %25-Aug-2018(10.31Am)

@author: %A.Nishanth C00294860
"""

import os
import pickle
import numpy as np
import csv
#%% change directory and load the files
"""
go to the directory and find the relevant PDB files

"""

working_dir = "C:/Users/nishy/Documents/Studies in UL/Projects_UL/Continues BIBM/NN-Results/Tier_1_results"
os.chdir('/')
os.chdir(working_dir)
files = os.listdir()