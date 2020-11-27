# -*- coding: utf-8 -*-
"""
Created on %23-Feb-2019(12.10 P.m)

@author: %A.Nishanth C00294860

choose surface Cα  atoms choosen for threshold conditions
"""
import os
import pickle

thresh_hold_ca=7.2
thresh_hold_res=6.7

#%%
def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
#    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
#%%
name = "ONGO"
pdb_details = pdb_details_fun(name,loading_dir)
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
saving_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
#for i in  range(0,len(pdb_details)):
#    pdb_list = pdb_details[i]
#    for pdb_name in pdb_list:
#        if pdb_name!='4MDQ':
##            print(pdb_name, ' working')
#            pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir)
pdb_name='121p'

#%%
#def pikle_create_MOTIF_depth_stats(pdb_name,saving_dir,loading_pikle_dir):
"""
This function create a pikle files for MOTIF depths
"""
os.chdir('/')
os.chdir(loading_pikle_dir)
coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))         

coordinates_selected=[]
corresponding_aminoacids=[]
for i in range(0,len(fin_res_depth_all)):
    if fin_ca_depth_all[i] <= thresh_hold_ca:
        if fin_res_depth_all[i] <= thresh_hold_res:
            coordinates_selected.append(coordinates[i])
            corresponding_aminoacids.append(aminoacids[i])
#%%
c_alpha_indexes_MOTIF= sum(MOTIF_indexs_all, [])

#os.chdir('/')
#os.chdir(saving_dir)
#pickle.dump(depth_ca_temp, open( ''.join(["depth_ca_temp_",pdb_name,".p"]), "wb" ) ) 
#pickle.dump(depth_res_temp, open( ''.join(["depth_res_temp_",pdb_name,".p"]), "wb" ) ) 
        
