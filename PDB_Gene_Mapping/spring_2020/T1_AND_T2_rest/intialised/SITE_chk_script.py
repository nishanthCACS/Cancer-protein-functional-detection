# -*- coding: utf-8 -*-
"""
Created on  %(26-Apr-2020) 5.03Pm

@author: %A.Nishanth C00294860

Little modified version 
%(09-Sep-2018) 6.07Pm
to check the SITE information to gether those PDBS satisfied Threhold condition
"""

import os
#from SITE_chk_class import SITE_check
#
#checking_dir_PDB = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/2020_PDBs_needed'
#PDBs =os.listdir(checking_dir_PDB)
#
#SITE_check(PDBs,checking_dir_PDB)
#%% Then go through the selected SITE
import os
import pickle
from copy import deepcopy

checking_dir_PDB = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/2020_PDBs_needed'
os.chdir('/')
os.chdir(checking_dir_PDB)
New_all_SITE_satisfied= pickle.load(open('New_all_SITE_satisfied.p', "rb"))        
New_all_SITE_satisfied_experiment_type= pickle.load(open('New_all_SITE_satisfied_experiment_type.p', "rb"))        
for i in range(0,len(New_all_SITE_satisfied_experiment_type)):
    if New_all_SITE_satisfied_experiment_type[i]!='SOFTWARE                                              ':
        print(New_all_SITE_satisfied_experiment_type[i])
        print(New_all_SITE_satisfied[i])
        
#%% Then create the genes with new SITE satisfied LIST
"""
load the other SITE satisied lists as well    
checking_dir_PDB_1

checking_dir_PDB_2: definitely go through MSMS
"""
checking_dir_PDB_1='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked'
checking_dir_PDB_2='C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results'

def SITE_sat_PDB_retrieve(directory):
    os.chdir('/')
    os.chdir(directory)
    names_whole=os.listdir(directory)
    SITE_satisfied_temp=[]
    for n in names_whole:
        if len(n)>17:
            if n[-16:-2]=='SITE_satisfied':
                SITE_satisfied_t= pickle.load(open(n, "rb"))        
                SITE_satisfied_t=sum(SITE_satisfied_t,[])
                SITE_satisfied_temp.append(deepcopy(SITE_satisfied_t))
    SITE_satisfied_temp =sum(SITE_satisfied_temp,[])
    SITE_satisfied_temp= list(set(SITE_satisfied_temp)) 
    return SITE_satisfied_temp

SITE_satisfied_temp_1=SITE_sat_PDB_retrieve(checking_dir_PDB_1)
SITE_satisfied_temp_2=SITE_sat_PDB_retrieve(checking_dir_PDB_2)

New_SITE_PDBs=[]
for n in New_all_SITE_satisfied:
    New_SITE_PDBs.append(n[0:-4])
SITE_sat_all=SITE_satisfied_temp_1+SITE_satisfied_temp_2+New_SITE_PDBs
SITE_sat_all= list(set(SITE_sat_all)) 

SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
os.chdir('/')
os.chdir(SITE_dir)
pickle.dump(SITE_sat_all, open("SITE_sat_all_T1_T2_2020.p", "wb" ))      
#%%
'''
Then go through the SITE satisfied PDBs to find the relation between T_1 and T_2
'''
import os
import pickle
from copy import deepcopy

working_dir_part="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"
SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'

os.chdir('/')
os.chdir(SITE_dir)
SITE_satisfied_all= pickle.load(open('SITE_sat_all_T1_T2_2020.p', "rb"))        
#%%
'''
load the 
    _Ids_info_all
then match them and form the SITE satisfied PDB groups for them
'''
def SITE_staisfy_creator(name,threshold_length=81):

    Ids_info_all = pickle.load(open( ''.join([name,"_Ids_info_all.p"]), "rb" ) )
    PDBs_sel=[]
    resolution=[]
    for ids in Ids_info_all:
        PDBs_sel_t=[]
        resolution_t=[]
        for id_t in ids:
            if len(id_t) == 3:
                resolution_t.append(float(id_t[1]))
                for j in range(0,len(id_t[2])):
                        if id_t[2][j]=="-":
                           if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > threshold_length:
                               if id_t[0] in SITE_satisfied_all:
                                   PDBs_sel_t.append(id_t[0])   
        PDBs_sel.append(deepcopy(PDBs_sel_t))
        resolution.append(deepcopy(resolution_t))
    pickle.dump(PDBs_sel, open(''.join([name,"_SITE_satisfied.p"]), "wb" ))      
    pickle.dump(resolution, open(''.join([name,"_SITE_satisfied_resolution.p"]), "wb" ))   

for Tier in [1,2]:
    loading_dir=''.join([working_dir_part,'Tier_',str(Tier),'_pdb_pikles_length_gene'])
    
    names_whole =os.listdir(loading_dir)
    sets=[]
    os.chdir('/')
    os.chdir(loading_dir)
    for n in names_whole:
        if n[-6:-3]=='PDB':
            if  n[0:-7] not in sets:
                sets.append(n[0:-7])
                SITE_staisfy_creator(n[0:-7])
##%%
#Tier=1
#loading_dir=''.join([working_dir_part,'Tier_',str(Tier),'_pdb_pikles_length_gene'])
#os.chdir('/')
#os.chdir(loading_dir)
#name='ONGO'
#PDBs_sel= pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
#%% 
'''
To create the PDB list that have to go through the 10-fold models
''' 
PDBs_needed_to_feed_through_the_models=[]               
for pdb in SITE_satisfied_all:
    if pdb not in SITE_satisfied_temp_2:
        PDBs_needed_to_feed_through_the_models.append(pdb)
pickle.dump(PDBs_needed_to_feed_through_the_models, open("SITE_MSMS_quater_needed.p", "wb" ))   
