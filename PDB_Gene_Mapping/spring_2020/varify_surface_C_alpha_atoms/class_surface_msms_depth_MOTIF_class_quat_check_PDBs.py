# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import os
import pickle
import math
import numpy as np
from copy import deepcopy 

class surface_msms_depth_MOTIF_class_quat_check_PDBs:
    """
    USE MSMS tool(1996) to calculte the soluable access area of the surface
        depth of the C_alpha carbons, and 
              of the residue 
                          from the surface
    this vertion fix the principle direction
    """
    def __init__(self, loading_pikle_dir, saving_dir, pdb_name,height=200,negation_included_all=False,w_o_mass=False,assign_prop_round_negation_w_avg_mass_chk=False):
        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class
        self.w_o_mass = w_o_mass#this neglet the "Mass spectrometry" property
        self.negation_included_all=negation_included_all
        self.assign_prop_round_negation_w_avg_mass_chk=assign_prop_round_negation_w_avg_mass_chk

        
        '''Assign property way'''
        self.old_17_prop=True
        #to select the optimal direction it should be true
        self.optimal_tilt=True
        '''loading_the_thresholds
        Earlier thresh_hold_ca=7.2,thresh_hold_res=6.7
        '''
        loading_dir_threshold = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
#         
        os.chdir("/")
        os.chdir(loading_dir_threshold)
#        thresh_hold_ca= pickle.load(open("max_depths_ca_MOTIF.p", "rb"))  
#        thresh_hold_res=pickle.load(open("max_depths_res_MOTIF.p", "rb"))  
        thresh_hold_ca=7.2
        thresh_hold_res=6.7

        os.chdir('/')
        os.chdir(loading_pikle_dir)
        coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
        aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
        fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
        fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
        MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))         
        
        c_alpha_indexes_MOTIF= sum(MOTIF_indexs_all, [])
        res_factor = 2.25 # see the documentation twhy 2.25 is chosen

        sur_res = []
        sur_res_cor_intial = []
        MOTIF_prop =[]
        
        #% to find out the surface atoms residues 
        for i in range(0,len(fin_res_depth_all)):
            if fin_ca_depth_all[i] <= thresh_hold_ca:
                if fin_res_depth_all[i] <= thresh_hold_res:
                    sur_res.append(aminoacids[i])
                    # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
                    #sur_res_cor_intial_round.append([round(res_factor*coordinates[i][0]),round(res_factor*coordinates[i][1]),round(res_factor*coordinates[i][2])])
                    sur_res_cor_intial.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])

                    if i in c_alpha_indexes_MOTIF:
                        MOTIF_prop.append(1)
                    else:
                        MOTIF_prop.append(0)
                        
        if len(sur_res_cor_intial)==0:       
            print(pdb_name, ' not has single atom to satify')