# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

import os
import csv
import pickle
from class_msms_surface  import msms_surface_MOTIF_class

clean_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/"
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/"
SITE_MUST = True
clean=False

name = "TSG"
saving_dir = "".join([working_dir, "/After_sucess/SITE_MUST_",name])

#% create the surface atoms through the MSMS
os.chdir('/')
os.chdir(site_dir)
pdb_details = pickle.load(open( ''.join([name,"_overlapped_O_T.p"]), "rb" ))
pdb_details=pdb_details[0]


pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#pdb_name='721p'
for i in range(0,len(pdb_details)):
    pdb_name = pdb_details[i]
    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
    msms_obj.retrieve_info()
    del msms_obj
	
	
