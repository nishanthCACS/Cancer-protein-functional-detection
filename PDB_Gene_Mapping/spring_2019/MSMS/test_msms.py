# -*- coding: utf-8 -*-
"""
Created on %17-Feb-2019(12.10 P.m)

@author: %A.Nishanth C00294860
"""
import os
import pickle

from msms_surface_MOTIF_class  import msms_surface_MOTIF_class

def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
#    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

##%% For Tier_1
#name = "ONGO"
#
#saving_dir = ''.join([saving_dir_part,name])
#pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#
#pdb_details = pdb_details_fun(name,loading_dir)
#print("considering: ",pdb_details[9][0])
##pdb_name = pdb_details[9][0]
#problemtric_pdbs=["6AMB","5YFN","3GFT","5USJ","5WLB","3HHM","4OVV","5SW8","5SWG","5SWO","5SWP","5SWR","5SWT","5SX8","5SX9","5SXA","5SXB","5SXC","5SXD","5SXE","5SXF","5SXI","5SXJ","5SXK","5FI0","5IFE","4JKV","4N4W","5DIS"]
#for pdb_name in problemtric_pdbs:
#    pdb_file = ''.join([pdb_name,".pdb"])
#    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#    msms_obj.retrieve_info()
#    del msms_obj
#for pdb_list in pdb_details:
#    for pdb_name in pdb_list:
#        pdb_file = ''.join([pdb_name,".pdb"])
#                       
#        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#        msms_obj.retrieve_info()
#        del msms_obj
#%%
name = "TSG"

saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])

pdb_details = pdb_details_fun(name,loading_dir)
#print("considering: ",pdb_details[9][0])
#pdb_name = pdb_details[9][0]
problemtric_pdbs=["3Q4T","3SOC","4ASX","1YPZ","2F53","5KNM","1O6S","2OMT","2OMU","2OMV","2OMX","2OMY","3HHM","4OVV","5SW8","5SWG","5SWO","5SWP","5SWR","5SWT","5SX8","5SX9","5SXA","5SXB","5SXC","5SXD","5SXE","5SXF","5SXI","5SXJ","5SXK","5GLJ"]
for pdb_name in problemtric_pdbs:
    pdb_file = ''.join([pdb_name,".pdb"])
    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
    msms_obj.retrieve_info()
    del msms_obj
#
#%%
name = "Fusion"
saving_dir = ''.join([saving_dir_part,name])
pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])

pdb_details = pdb_details_fun(name,loading_dir)
#print("considering: ",pdb_details[9][0])
#pdb_name = pdb_details[9][0]
problemtric_pdbs=["5UC4","5UCH","5UCI","5UCJ","5DIS"]
for pdb_name in problemtric_pdbs:
    pdb_file = ''.join([pdb_name,".pdb"])
    msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
    msms_obj.retrieve_info()
    del msms_obj
#os.chdir('/')
#os.chdir(saving_dir)
#
#MOTIF_indexs_all = pickle.load(open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))
#coordinates=  pickle.load(open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
#amino_acid= pickle.load(open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
#fin_res_depth_all=pickle.load(open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
#fin_ca_depth_all= pickle.load(open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
#surface = np.load(''.join(['surface_',pdb_name,'.npy']))
#%%
#name = "TSG"
#
#saving_dir = ''.join([saving_dir_part,name])
#pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#pdb_details = pdb_details_fun(name,loading_dir)
#
#for pdb_list in pdb_details:
#    for pdb_name in pdb_list:
#        pdb_file = ''.join([pdb_name,".pdb"])
#                       
#        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#        msms_obj.retrieve_info()
#        del msms_obj
#    
#name = "Fusion"
#
#saving_dir = ''.join([saving_dir_part,name])
#pdb_source_dir = ''.join([pdb_source_dir_part, name,'_R'])
#pdb_details = pdb_details_fun(name,loading_dir)
#
#for pdb_list in pdb_details:
#    for pdb_name in pdb_list:
#        pdb_file = ''.join([pdb_name,".pdb"])
#                       
#        msms_obj = msms_surface_MOTIF_class(pdb_source_dir, saving_dir, pdb_name)
#        msms_obj.retrieve_info()
#        del msms_obj
#