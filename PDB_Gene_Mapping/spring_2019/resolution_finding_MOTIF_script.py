# -*- coding: utf-8 -*-
"""
Created on %25-January-2018(1.08pm)


@author: %A.Nishanth C00294860
loading the experiment type details
"""
import os
import pickle
from resolution_finding_MOTIF_class  import resolution_finding_MOTIF

def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
#    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details
#%% For Tier_1
name = "ONGO"
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_source_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/ONGO_R"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/ONGO"
pdb_details = pdb_details_fun(name,loading_dir)
for pdb_list in pdb_details:
    for pdb_name in pdb_list:
        if len(pdb_name)>2:
#            print("considering: ",pdb_details[0][0])
#            pdb_name = pdb_details[0][0]
            OG = resolution_finding_MOTIF(pdb_source_dir,saving_dir,pdb_name)
            OG.resolution_mumpy_pikle()
            OG.resolution_per_mumpy_pikle()
#%%
name = "TSG"
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_source_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/TSG_R"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/TSG"
pdb_details = pdb_details_fun(name,loading_dir)
for pdb_list in pdb_details:
    for pdb_name in pdb_list:
        if len(pdb_name)>2:
#            print("considering: ",pdb_details[0][0])
#            pdb_name = pdb_details[0][0]
            TSG = resolution_finding_MOTIF(pdb_source_dir,saving_dir,pdb_name)
            TSG.resolution_mumpy_pikle()
            TSG.resolution_per_mumpy_pikle()
#%
name = "Fusion"
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_source_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/Fusion_R"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/Fusion"
pdb_details = pdb_details_fun(name,loading_dir)
for pdb_list in pdb_details:
    for pdb_name in pdb_list:
        if len(pdb_name)>2:
#            print("considering: ",pdb_details[0][0])
#            pdb_name = pdb_details[0][0]
            Fusion = resolution_finding_MOTIF(pdb_source_dir,saving_dir,pdb_name)
            Fusion.resolution_mumpy_pikle()
            Fusion.resolution_per_mumpy_pikle()
#%%
"""
Then load the pikle files and addup them and make the ovelall matrix
"""
import numpy as np

def load_np_pikles(saving_dir):
    """
    this functin load the numpy pikles and addthem up and retuen the sum of the # missed MOTIFs
    """
    os.chdir(saving_dir)
    pikles = os.listdir()
    missed= np.zeros((6,5))
    for p in pikles:
        if "per" not in p:
            missed = missed + np.load(p)
    return missed

saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/ONGO"
missed_OG = load_np_pikles(saving_dir)
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/TSG"
missed_TSG = load_np_pikles(saving_dir)    
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/Fusion"
missed_Fusion = load_np_pikles(saving_dir)
#%% Total missef MOTIFs
missed= np.zeros((6,5))
missed = missed_OG + missed_TSG + missed_Fusion
#%%
def load_np_pikles_per(saving_dir):
    """
    this functin load the numpy pikles of percentage
     and retuen the average persentage missed
    """
    os.chdir(saving_dir)
    pikles = os.listdir()
    missed= np.zeros((6,5))
    count = 0
    for p in pikles:
        if "per" in p:
            missed = missed + np.load(p)
            count = count +1 
    missed = missed/count
    return missed

saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/ONGO"
missed_per_OG = load_np_pikles_per(saving_dir)
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/TSG"
missed_per_TSG = load_np_pikles_per(saving_dir)    
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/Fusion"
missed_per_Fusion = load_np_pikles_per(saving_dir)

missed_per_all = (missed_per_OG + missed_per_TSG + missed_per_Fusion)/3