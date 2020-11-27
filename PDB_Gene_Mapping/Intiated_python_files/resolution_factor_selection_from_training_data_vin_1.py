# -*- coding: utf-8 -*-
"""
Created on %(18-Sep-2018) 14.45Pm

@author: %A.Nishanth C00294860
"""
import pickle
import os
import csv
#%%
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"


# this consider all the pdbs with satisfy the threshold_condition and X-ray diffraction
def load_the_PDBs_staisfy_basic_cond(name):
    """load all pdbs"""
    os.chdir('/')
    os.chdir(site_dir)
    fin_PDB_ids = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))    
    all_pdbs = sum(fin_PDB_ids, [])
    
    resolutions = []
    loading_pdb_Source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
   
    os.chdir('/')
    os.chdir(loading_pdb_Source_dir)
    #% then go through the PDB_ids and take the resolution details
    for pdb_id in all_pdbs:
        pbd_name = ''.join([pdb_id,".pdb"])
        f = open(pbd_name)
        csv_f = csv.reader(f)
        stack=[]
        for row in csv_f:
            for s in row:
                stack.append(s)
        
        for row in stack:
            temp = row.split()
            if row[0:21] == 'REMARK   2 RESOLUTION':
                # to access the resolution details
                resolution_PDB = float(temp[3])
        #        print("Resolution: ", float(temp[3]))
        resolutions.append(resolution_PDB)
    print("--------", name, "--------")
    print("number_of_PDBs       : ",len(all_pdbs))
    print("number_of_resolutions: ",len(resolutions))
    print("average_resolution   : ",sum(resolutions)/len(resolutions))
    return all_pdbs, resolutions

ONGO_pdbs, ONGO_resolutions = load_the_PDBs_staisfy_basic_cond("ONGO")
TSG_pdbs, TSG_resolutions = load_the_PDBs_staisfy_basic_cond("TSG")
Fusion_pdbs,Fusion_resolutions = load_the_PDBs_staisfy_basic_cond("Fusion")

#%% 
all_pdbs =  ONGO_pdbs + TSG_pdbs + Fusion_pdbs
all_resolutions = ONGO_resolutions + TSG_resolutions + Fusion_resolutions
